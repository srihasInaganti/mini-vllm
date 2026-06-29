# block manager invariants: allocate/free, no double-free, no leak, reuse after a sequence ends
import pytest

from engine.block_manager import BlockManager, BlockTable


def test_allocate_returns_distinct_ids_and_shrinks_pool():
    m = BlockManager(num_blocks=8, block_size=16)
    ids = m.allocate(3)
    assert len(set(ids)) == 3        # no id handed out twice
    assert m.num_free == 5


def test_free_restores_the_pool():
    m = BlockManager(num_blocks=4, block_size=16)
    ids = m.allocate(4)
    assert m.num_free == 0
    m.free(ids)
    assert m.num_free == 4           # no leak: every block came back


def test_double_free_is_caught():
    m = BlockManager(num_blocks=4, block_size=16)
    ids = m.allocate(2)
    m.free(ids)
    with pytest.raises(AssertionError):
        m.free(ids)                  # second free of the same blocks must blow up


def test_allocating_more_than_available_raises():
    m = BlockManager(num_blocks=2, block_size=16)
    with pytest.raises(RuntimeError):
        m.allocate(3)


def test_block_table_allocates_only_when_a_block_fills():
    m = BlockManager(num_blocks=8, block_size=16)
    bt = BlockTable(m)
    bt.append(16)                    # exactly one block
    assert len(bt.blocks) == 1
    bt.append(1)                     # spills into a second block
    assert len(bt.blocks) == 2
    assert bt.length == 17


def test_block_table_slot_mapping():
    m = BlockManager(num_blocks=8, block_size=16)
    bt = BlockTable(m)
    bt.append(20)                    # two blocks
    b0, b1 = bt.blocks
    assert bt.slot(0) == b0 * 16 + 0
    assert bt.slot(15) == b0 * 16 + 15
    assert bt.slot(16) == b1 * 16 + 0    # first token of the second block
    assert bt.all_slots() == [bt.slot(p) for p in range(20)]


def test_blocks_are_reusable_after_a_sequence_ends():
    m = BlockManager(num_blocks=4, block_size=16)
    bt = BlockTable(m)
    bt.append(40)                    # three blocks
    assert m.num_free == 1
    bt.free()
    assert m.num_free == 4           # freed back
    # a fresh sequence can now reuse the whole pool
    bt2 = BlockTable(m)
    bt2.append(64)
    assert len(bt2.blocks) == 4
    assert m.num_free == 0
