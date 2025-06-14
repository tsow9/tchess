

class EvalCache:
  def __init__(self):
    self.cache = {}
    self.hit = 0
    self.miss = 0

  def get(self, board_hash_key: int):
    if board_hash_key in self.cache:
      self.hit += 1
    else:
      self.miss += 1
    return self.cache.get(board_hash_key, None)

  def set(self, board_hash_key: int, score: float):
    self.cache[board_hash_key] = score

  def log_stats(self):
    total = self.hit + self.miss
    hit_rate = self.hit / total * 100 if total > 0 else 0
    print(f"EvalCache hits: {self.hit}, misses: {self.miss}, hit rate: {hit_rate:.2f}%")