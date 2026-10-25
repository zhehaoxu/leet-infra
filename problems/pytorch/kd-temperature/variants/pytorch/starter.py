import torch


def solve(student_logits, teacher_logits, T, reduction="mean"):
    """知识蒸馏损失：T^2 * KL(softmax(teacher/T) || softmax(student/T))，逐样本后
    按 reduction mean/sum/none 归约。非法 reduction 抛 ValueError。"""
    # TODO
    raise NotImplementedError
