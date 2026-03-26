# 篮球上篮测试系统 - 状态机逻辑文档

1. 测试目标
记录学生完成4个有效进球的总时间
每个有效进球之间必须出三分线
每轮尝试包含：1次首次上篮 + 最多2次补篮机会
如果连续两次补篮未进，本轮作废，需要出三分线重新开始
一轮尝试结束后必须出三分线才能开始下一轮

2. 单次进球流程
上篮尝试
进球 -> 计入有效进球数，必须出三分线
未进球 -> 进入补篮阶段
补篮阶段（如果上篮未进）
补篮进球 -> 计入有效进球数，必须出三分线
补篮未进 -> 可以再补一次
连续两次补篮未进 -> 本轮作废，必须出三分线重新开始
出三分线
每个有效进球后必须出三分线
连续两次补篮未进后必须出三分线
不出三分线的进球无效

3. 状态定义
class State(Enum):
    INIT = auto()        # 初始状态，等待开始
    FIRST_SHOT = auto()  # 首次上篮阶段
    RETRY_SHOT = auto()  # 补篮阶段
    OUT_3PT = auto()     # 出三分阶段
    COMPLETE = auto()    # 完成状态（4个有效进球）

4. 需要跟踪的计数
有效进球数 (valid_shots: 0-4)
当前补篮次数 (retry_count: 0-2)
当前轮次是否已出三分 (can_start_new: bool)

5. 状态转换序列示例
INIT ->
FIRST_SHOT[未进] -> RETRY_SHOT[未进] -> RETRY_SHOT[未进] -> OUT_3PT ->  # 本轮作废
FIRST_SHOT[进球] -> OUT_3PT ->  # 有效进球+1
FIRST_SHOT[未进] -> RETRY_SHOT[进球] -> OUT_3PT ->  # 有效进球+1
FIRST_SHOT[未进] -> RETRY_SHOT[未进] -> RETRY_SHOT[进球] -> OUT_3PT ->  # 有效进球+1
FIRST_SHOT[进球] -> OUT_3PT ->  # 有效进球+1
COMPLETE

6. 事件类型
class Event(Enum):
    FIRST_SHOT_MADE = "first_shot_made"    # 首次上篮进球
    FIRST_SHOT_MISSED = "first_shot_missed" # 首次上篮未进
    RETRY_SHOT_MADE = "retry_shot_made"     # 补篮进球
    RETRY_SHOT_MISSED = "retry_shot_missed" # 补篮未进
    OUT_3PT = "out_three_point"            # 出三分
    START = "start"                        # 开始计时
    END = "end"                           # 结束计时

7. 状态转换规则
INIT状态
接收START事件 -> 转到FIRST_SHOT状态，开始第一轮尝试

FIRST_SHOT状态
接收FIRST_SHOT_MADE事件：
有效进球数+1
转到OUT_3PT状态
如果有效进球数达到4，转到COMPLETE状态
接收FIRST_SHOT_MISSED事件：
转到RETRY_SHOT状态
补篮次数设为0

RETRY_SHOT状态
接收RETRY_SHOT_MADE事件：
有效进球数+1
转到OUT_3PT状态
如果有效进球数达到4，转到COMPLETE状态
接收RETRY_SHOT_MISSED事件：
补篮次数+1
如果补篮次数达到2次：
转到OUT_3PT状态（本轮作废）
否则保持RETRY_SHOT状态

OUT_3PT状态
接收OUT_3PT事件：
如果有效进球数达到4：
转到COMPLETE状态
否则：
设置can_start_new=true
转到FIRST_SHOT状态，可以开始新一轮尝试

COMPLETE状态
终止状态，不再接受事件

记录的数据

、、、python
{
    "timestamp": float,          # 事件时间戳
    "event": str,               # 事件类型
    "from_state": str,          # 转换前状态
    "to_state": str,            # 转换后状态
    "valid_shots": int,         # 有效进球数(0-4)
    "retry_count": int,         # 当前补篮次数(0-2)
    "can_start_new": bool,      # 是否可以开始新一轮（已出三分）
    "is_complete": bool         # 是否完成测试
}
、、、
