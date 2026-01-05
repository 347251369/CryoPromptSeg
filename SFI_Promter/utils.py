import random
import torch.distributed as dist

import numpy as np
import scipy.spatial as S
import torchvision.transforms as T

import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
      # 用于存储不同的指标（如 loss、accuracy 等），其中 SmoothedValue 是一个用于平滑数值变化的类（一般用于计算滑动平均值）。
        self.meters = defaultdict(SmoothedValue)
        # 日志信息的分隔符，默认为 \t（制表符），传入的是"  "
        self.delimiter = delimiter

    # 传入一组关键字参数（如 loss=0.5, accuracy=0.9）
    def update(self, **kwargs):
      # 如果值是 torch.Tensor，转换为 Python 标量（item()）
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            # 使用 SmoothedValue.update(v) 更新相应的指标
            self.meters[k].update(v)

    # 允许 MetricLogger 通过 metric_logger.loss 直接访问 self.meters['loss']
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    # 遍历所有已记录的指标，将其格式化为字符串
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    # 在分布式训练中，同步不同进程的指标，确保日志在不同设备之间保持一致。
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    # 手动向 self.meters 添加一个指标（meter）
    def add_meter(self, name, meter):
        self.meters[name] = meter

    # 可迭代对象（通常是 dataloader）
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        # iter_time 计算单次迭代时间的平均值
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        # data_time 计算数据加载时间的平均值
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                # 计算预计剩余时间（ETA）
                    "eta: {eta}",  
                    "{meters}",
                    # 计算迭代时间
                    "time: {time}",
                    # 计算数据加载时间
                    "data: {data}",
                    # 记录 GPU 显存占用
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end) # 计算数据加载时间
            yield obj
            iter_time.update(time.time() - end)# 计算迭代时间
            # 每 print_freq 步或最后一步打印日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                # 计算预计剩余时间（ETA）
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        # 计算整个过程的时间，并打印每个 batch 的平均时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# 如果分布式环境不可用或未初始化，则返回1，否则返回实际的世界大小。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def get_tp(
        pred_points,
        pred_scores,
        gd_points,
        thr=12,
        return_index=False
):
    sorted_pred_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_pred_indices]

    unmatched = np.ones(len(gd_points), dtype=bool)
    dis = S.distance_matrix(sorted_pred_points, gd_points)

    for i in range(len(pred_points)):
        min_index = dis[i, unmatched].argmin()
        if dis[i, unmatched][min_index] <= thr:
            unmatched[np.where(unmatched)[0][min_index]] = False

        if not np.any(unmatched):
            break

    if return_index:
        return sum(~unmatched), np.where(unmatched)[0]
    else:
        return sum(~unmatched)

# 抑制距离过近且得分较低的点
# nms_thr 参数控制了两个点之间的最大允许距离，只有当两个点的距离大于此阈值时，才会同时保留
def point_nms(points, scores, classes, nms_thr=-1):
  # 创建一个布尔数组 _reserved，初始值全部为 True，表示所有点都保留。
    _reserved = np.ones(len(points), dtype=bool)
    # 计算点之间的距离矩阵 dis_matrix，并将对角线元素设为无穷大，避免自身比较。
    dis_matrix = S.distance_matrix(points, points)
    np.fill_diagonal(dis_matrix, np.inf)

    for idx in np.argsort(-scores):
        if _reserved[idx]:
            _reserved[dis_matrix[idx] <= nms_thr] = False

    points = points[_reserved]
    scores = scores[_reserved]
    classes = classes[_reserved]

    return points, scores, classes


def set_seed(args):
    seed = args.seed
    # seed = args.seed + get_rank()

    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pre_processing(img):
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img).unsqueeze(0)


@torch.no_grad()
def predict(
        model,
        img_partA,
        nms_thr=-1,
        ori_shape=None,
        filtering=False
):
    ori_h, ori_w = ori_shape
    outputs, unet_y = model(img_partA)
    
    
    # 从模型输出中获取预测的坐标信息
    points = outputs['pred_coords'][0].cpu().numpy()
    # 从模型输出中获取预测的未归一化的对数概率（logits）。将其转换为概率分布（每个类别的置信度分数）。
    scores = outputs['pred_logits'][0].softmax(-1).cpu().numpy()
    # 在最后一维（即类别维度）上找到最大值的索引，表示预测的类别标签。
    classes = np.argmax(scores, axis=-1)

    
    # 保证坐标在图像内
    np.clip(points[:, 0], a_min=0, a_max=ori_w - 1, out=points[:, 0])
    np.clip(points[:, 1], a_min=0, a_max=ori_h - 1, out=points[:, 1])
    # 过滤无效点：移除类别为背景的点（即 classes 小于 scores.shape[-1] - 1 的点）。
    valid_flag = classes < (scores.shape[-1] - 1)


    points = points[valid_flag]
    scores = scores[valid_flag].max(1)
    classes = classes[valid_flag]

    mask = outputs['pred_masks'][0, 0].cpu().numpy() > 0
# 通过mask过滤掉在错误点
    if filtering:
        valid_flag = mask[points.astype(int)[:, 1], points.astype(int)[:, 0]]
        points = points[valid_flag]
        scores = scores[valid_flag]
        classes = classes[valid_flag]

    if len(points) and nms_thr > 0:
        points, scores, classes = point_nms(points, scores, classes, nms_thr)

    return points, scores, classes, mask , unet_y


def collate_fn(batch):
    img_partA, img_partB, points, labels, masks = [[] for _ in range(5)]
    for x in batch:
        img_partA.append(x[0])
        img_partB.append(x[1])
        points.append(x[2])
        labels.append(x[3])
        masks.append(x[4])
    # 这个新的张量的形状将是 (batch_size, *image_shape)
    return torch.stack(img_partA),torch.stack(img_partB) ,torch.stack(masks), points, labels

def quantize(x, mi=-3, ma=3, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x
