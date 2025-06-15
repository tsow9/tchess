
class EvalCache:
  def __init__(self):
    self.cache = {}
    self.hit = self.miss = 0

  def get(self, board_hash_key: int):
    if board_hash_key in self.cache:
      self.hit += 1
    else:
      self.miss += 1
    return self.cache.get(board_hash_key, None)   # Returns the score if the key is available, or None if not.

  def set(self, board_hash_key: int, score):
    self.cache[board_hash_key] = score

  def log_stats(self):
    total = self.hit + self.miss
    hit_rate = self.hit / total * 100 if total > 0 else 0
    print(f"EvalCache hits: {self.hit}, misses: {self.miss}, hit rate: {hit_rate:.2f}%")