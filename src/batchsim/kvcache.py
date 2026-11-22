"""Block-granular KV-cache accounting.

The simulator never needs block identities (prefix caching is out of scope), only how many
blocks each request holds and how many are free, so allocation is O(1).
"""

from __future__ import annotations

from batchsim.request import Request


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = num_blocks
        self.peak_used = 0

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def needed(self, req: Request, tokens: int) -> int:
        """Additional blocks ``req`` needs to hold ``tokens`` tokens."""
        return max(0, -(-tokens // self.block_size) - req.num_blocks)

    def allocate(self, req: Request, tokens: int, keep_free: int = 0) -> bool:
        """Grow ``req`` to hold ``tokens`` tokens, leaving ``keep_free`` blocks spare."""
        need = -(-tokens // self.block_size) - req.num_blocks
        if need <= 0:
            return True
        if need + keep_free > self.free:
            return False
        self.free -= need
        req.num_blocks += need
        used = self.num_blocks - self.free
        if used > self.peak_used:
            self.peak_used = used
        return True

    def release(self, req: Request) -> None:
        self.free += req.num_blocks
        req.num_blocks = 0

    @property
    def used(self) -> int:
        return self.num_blocks - self.free

    @property
    def usage(self) -> float:
        return self.used / self.num_blocks
