# AI接口消息格式规范

## 字段规范

### 顶层字段

- `message_id`: [必填] 信息唯一UUID
- `game_id`: [必填] 考试UUID（创建时通过命令行传入）
- `message_type`: [必填] 信息类型
- `message_timestamp`: [必填] 信息发送的时间戳
- `data`: [必填] 数据类型
  - events
  - summary
  - ...

### 事件字段

- `event_category`: [必填] 事件具体类别，根据event_type选择对应的枚举值
- `timestamp`: [必填] 事件发生时间，相对于视频开始的毫秒数，使用整形表示

## 事件类别枚举值定义

### 动作类别 (`event_type` = `action`)

- `layup`: 上篮
- `putback`: 补篮
- `out_three_point`: 出三分

### 违规类别 (`event_type` = `violation`)

- `traveling`: 走步
- `double_dribble`: 二运
- `stationary_layup`: 非行进间上篮
- `three_point_line`: 没出三分线
- `incomplete_shots`: 没有完成四次投篮
- `out_of_bounds`: 出界

### 普通事件类别 (`event_type` = `normal`)

- `start`: 开始
- `score`: 进球
- `end`: 结束

## 数据约束

1. 事件必须按`timestamp`时间顺序升序排列
2. 每个事件的`event_id`必须唯一
3. `event_type`和`event_category`必须匹配，使用上述定义的枚举值
4. 时间戳应当以秒为单位，精确到小数点后一位
5. 所有时间戳必须大于等于0，且不超过视频总时长
6. `summary`中的统计数据必须与实际`events`数组内容一致
