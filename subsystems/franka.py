"""
    Franka 机械臂通信文件
    分任务
"""

def grasp(pos, force, _time):
    """
    Order the Franka machine to grasp an object at pos=(x, y, z);
    Maintain a force for some seconds(_time)
    :param pos:     the object position
    :param force:   force num
    :param _time:   duration.
    :return:
    """
    