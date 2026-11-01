"""Bounded, lossless packed-expert LRU on the tensor-parallel mesh.

The host keeps the checkpoint. Only misses cross the host/device link; hits
are gathered on device. A working set larger than the cache is supported:
protect this call's hits, retain as many misses as fit, and stream the rest.
Cache updates donate their old buffers so residency does not grow over time.
"""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from glm53.engine import AXIS, R, shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .routing import expert_bucket


class ExpertCache:
    def __init__(self, store, mesh, transport, budget_bytes):
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("expert cache budget must be a nonnegative integer")
        self.store, self.mesh, self.transport = store, mesh, transport
        self.capacity, self.slots, self.buffers, self.programs = {}, {}, {}, {}
        # Divide the global budget equally between layers, including all chips.
        per_layer = budget_bytes // max(1, len(store.tables))
        reserved = 0
        for layer, tables in store.tables.items():
            per_expert = sum(a[:, :1].nbytes for a in tables)
            capacity = min(tables[0].shape[1], per_layer // per_expert)
            self.capacity[layer] = capacity
            reserved += capacity * per_expert
        self.stats = {
            "budget_bytes": budget_bytes,
            "reserved_bytes": reserved,
            "allocated_bytes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compact_table_bytes_avoided": 0,
            "capacity_by_layer": dict(self.capacity),
        }

    def clear_entries(self):
        """Start a diagnostic cache cycle without reallocating HBM or programs.

        Old bytes become unreachable; every newly referenced expert is loaded
        before use. Lifetime counters and the allocated buffers are retained.
        Call only between completed inference workloads.
        """
        for slots in self.slots.values():
            slots.clear()

    def _allocate(self, layer):
        capacity = self.capacity[layer]
        shapes = tuple(
            (a.shape[0], capacity, a.shape[2], a.shape[3] * a.shape[4])
            for a in self.store.tables[layer]
        )
        sharding = NamedSharding(self.mesh, P(AXIS))
        # Allocate zeros directly on the mesh, without a cache-sized host copy.
        self.buffers[layer] = jax.jit(
            lambda: tuple(jnp.zeros(shape, jnp.uint8) for shape in shapes),
            out_shardings=(sharding,) * 3,
        )()
        self.slots[layer] = OrderedDict()
        self.stats["allocated_bytes"] += sum(a.size for a in self.buffers[layer])

    def _program(self, layer, ep, miss_bucket, direct=False):
        shapes = tuple(a.shape for a in self.buffers[layer])
        key = (shapes, ep if not direct else None, miss_bucket, direct)
        if key in self.programs:
            return self.programs[key]

        def hit_only(cache, hit_slots):
            result = []
            for array in cache:
                selected = array[0][jnp.maximum(hit_slots, 0)]
                valid = (hit_slots >= 0).reshape((-1,) + (1,) * (selected.ndim - 1))
                result.append(jnp.where(valid, selected, 0)[None])
            return tuple(result)

        def update(cache, misses, hit_slots, miss_slots, destinations):
            staged, updated = [], []
            for array, raw in zip(cache, misses, strict=True):
                c, m = array[0], raw[0]
                hits = c[jnp.maximum(hit_slots, 0)]
                fresh = m[jnp.maximum(miss_slots, 0)]
                shape = (-1,) + (1,) * (hits.ndim - 1)
                selected = jnp.where((hit_slots >= 0).reshape(shape), hits, fresh)
                valid = ((hit_slots >= 0) | (miss_slots >= 0)).reshape(shape)
                staged.append(jnp.where(valid, selected, 0)[None])
                # Out-of-bounds destinations drop padding and unretained misses.
                updated.append(c.at[destinations].set(m, mode="drop")[None])
            return tuple(staged), tuple(updated)

        def update_direct(cache, misses, destinations):
            return tuple(
                array[0].at[destinations].set(raw[0], mode="drop")[None]
                for array, raw in zip(cache, misses, strict=True)
            )

        tables = (P(AXIS),) * 3
        if direct:
            program = jax.jit(
                shard_map(
                    update_direct,
                    mesh=self.mesh,
                    in_specs=(tables, tables, R),
                    out_specs=tables,
                    check_vma=False,
                ),
                donate_argnums=(0,),
            )
        elif miss_bucket:
            program = jax.jit(
                shard_map(
                    update,
                    mesh=self.mesh,
                    in_specs=(tables, tables, R, R, R),
                    out_specs=(tables, tables),
                    check_vma=False,
                ),
                donate_argnums=(0,),
            )
        else:
            program = jax.jit(
                shard_map(
                    hit_only,
                    mesh=self.mesh,
                    in_specs=(tables, R),
                    out_specs=tables,
                    check_vma=False,
                )
            )
        self.programs[key] = program
        return program

    def gather(self, layer, idx, *, direct=False):
        """Optionally return resident tables, miss tables and [cache, miss] slots.

        The direct view lets the expert kernel select only its current tile.
        It never builds a second full table of hits. Hits remain protected from
        eviction until this call's dependent expert computation has finished.
        """
        idx = np.asarray(idx)
        total = self.store.tables[layer][0].shape[1]
        if idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("integer expert IDs are required")
        if idx.min() < 0 or idx.max() >= total:
            raise ValueError("expert ID outside table")
        if not self.capacity[layer]:
            arrays, local, ep = self.store.gather(layer, idx)
            self.stats["misses"] += np.unique(idx).size
            return self.transport(arrays), local, ep
        if layer not in self.buffers:
            self._allocate(layer)
        selected, inverse = np.unique(idx, return_inverse=True)
        ep = expert_bucket(len(selected), total, self.store.tile)
        slots = self.slots[layer]
        hit_slots = np.full(ep, -1, np.int32)
        miss_slots = np.full(ep, -1, np.int32)
        missing = []
        protected = set()
        for row, expert in enumerate(selected.tolist()):
            if expert in slots:
                hit_slots[row] = slots[expert]
                protected.add(expert)
                slots.move_to_end(expert)
            else:
                miss_slots[row] = len(missing)
                missing.append(expert)
        self.stats["hits"] += len(protected)
        self.stats["misses"] += len(missing)
        if not missing:
            if direct:
                # No misses select this operand. Alias resident storage rather
                # than allocating or compiling another table of zeros.
                fresh = self.buffers[layer]
            else:
                packed = self._program(layer, ep, 0)(self.buffers[layer], hit_slots)
        else:
            raw, _, mp = self.store.gather(layer, np.asarray([missing], np.int32))
            capacity = self.capacity[layer]
            destinations = np.full(mp, capacity, np.int32)
            # Never evict an entry needed by this call. Over-capacity misses
            # still contribute to its output, even when they cannot be retained.
            victims = iter([expert for expert in slots if expert not in protected])
            for row, expert in enumerate(missing):
                if len(slots) < capacity:
                    slot = len(slots)
                else:
                    victim = next(victims, None)
                    if victim is None:
                        break
                    slot = slots.pop(victim)
                    self.stats["evictions"] += 1
                slots[expert] = slot
                destinations[row] = slot
            fresh = self.transport(raw)
            if direct:
                self.buffers[layer] = self._program(layer, ep, mp, direct=True)(
                    self.buffers[layer], fresh, destinations
                )
                retained = destinations[miss_slots.clip(0)]
                hit_slots = np.where(
                    (hit_slots < 0) & (miss_slots >= 0) & (retained < capacity),
                    retained,
                    hit_slots,
                ).astype(np.int32)
            else:
                packed, self.buffers[layer] = self._program(layer, ep, mp)(
                    self.buffers[layer], fresh, hit_slots, miss_slots, destinations
                )
        if direct:
            packed = (*self.buffers[layer], *fresh, np.stack((hit_slots, miss_slots), axis=1))
            self.stats["compact_table_bytes_avoided"] += ep * sum(
                a[:, :1].nbytes for a in self.store.tables[layer]
            )
        return packed, inverse.reshape(idx.shape).astype(np.int32), ep
