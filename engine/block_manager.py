# hands out fixed-size KV blocks from a shared pool and takes them back
# BlockTable maps one sequence's token positions to whatever physical blocks it got
from collections import deque


class BlockManager:
    """Owns the pool of physical blocks and the free-list of their ids."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = deque(range(num_blocks))
        # mirror of the free-list as a set, only so double-free is a cheap check
        self._free_set = set(range(num_blocks))

    def allocate(self, n: int = 1) -> list[int]:
        # running out is a real condition; for now it's an error (preemption comes later)
        if n > len(self._free):
            raise RuntimeError(f"out of KV blocks: need {n}, have {len(self._free)}")
        out = [self._free.popleft() for _ in range(n)]
        self._free_set.difference_update(out)
        return out

    def free(self, block_ids: list[int]) -> None:
        for b in block_ids:
            # freeing the same block twice would corrupt the pool, so catch it loudly
            assert b not in self._free_set, f"double free of block {b}"
            self._free.append(b)
            self._free_set.add(b)

    @property
    def num_free(self) -> int:
        return len(self._free)


class BlockTable:
    """One sequence's logical -> physical map: an ordered list of block ids plus length."""

    def __init__(self, manager: BlockManager):
        self.manager = manager
        self.block_size = manager.block_size
        self.blocks: list[int] = []   # physical block ids in logical order
        self.length = 0               # tokens stored so far

    def append(self, num_tokens: int) -> None:
        # grow the block list to cover the new length, allocating only when a block fills
        new_length = self.length + num_tokens
        needed = (new_length + self.block_size - 1) // self.block_size
        if needed > len(self.blocks):
            self.blocks += self.manager.allocate(needed - len(self.blocks))
        self.length = new_length

    def slot(self, pos: int) -> int:
        # flat address of a token: which block it lives in, times block_size, plus its offset
        block = self.blocks[pos // self.block_size]
        return block * self.block_size + pos % self.block_size

    def slots(self, positions) -> list[int]:
        return [self.slot(p) for p in positions]

    def all_slots(self) -> list[int]:
        return [self.slot(p) for p in range(self.length)]

    def free(self) -> None:
        self.manager.free(self.blocks)
        self.blocks = []
        self.length = 0
