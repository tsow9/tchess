import torch

def hook_clone_counter():
  real_clone = torch.Tensor.clone  # 元のclone関数を保存
  count = {'clone': 0}             # 呼び出しカウンター

  def counted_clone(self, *args, **kwargs):
    count['clone'] += 1
    return real_clone(self, *args, **kwargs)

  torch.Tensor.clone = counted_clone  # フックを適用
  return count
