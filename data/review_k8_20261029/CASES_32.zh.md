# 逐图病例

Codex 单一复核者；定向选样，不是总体错误率。原图与全部执行框均查看；小字处补看 env.present 图。视图为原区域重建，不声称恢复了实际处理器 tensor。E1 属于评测，任何正例都不直接加入 SFT。

## C01 · test_History_239

[结构化轨迹](C01.json)

**k8_dapo / sample 0**：supported；gold=J，pred=J，crop=2。

原图明确标 3100 BC 与 1250 BC；两次裁图分别覆盖左、右日期，差 1850 与 J 一致。首轮已陈述两日期，sample 2 同题直答也正确，因此这例有继续确认与直答的不同路径；不能仅凭此判两次调用必然无用。

类别：日期读取、跨视图结合、停止。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_dapo / sample 2**：supported；gold=J，pred=J，crop=0。

零裁图直接读取 3100、1250 并得到 1850，与原图和 J 一致。属于可核对的直答实例，不代表该组已学会普遍按需停止。

类别：证据充分直接作答。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C02 · mme_10450

[结构化轨迹](C02.json)

**k8_dapo / sample 0**：supported；gold=D，pred=D，crop=1。

obs1 的蓝柱为 Revenue：212、1378、4060、8170、13848，2024 首次超过 10000，D 正确。黄色折线为 Deals # Break Even level；未将它混入 Revenue。

类别：系列绑定、有效读取。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_grpo / sample 0**：refuted；gold=D，pred=A，crop=3。

把黄色 Deals # Break Even level 的数（11974、13142 等）说成蓝柱 Revenue，后续裁到别的情景表仍未改正；最后说最早 2020 并称对应 A，但 A 实为 2021。既有系列绑定错误，也有终答映射错误。

类别：L3 系列绑定；L4 未修正；L5 选项映射。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C03 · mme_18106

[结构化轨迹](C03.json)

**k8_dapo / sample 0**：supported；gold=B，pred=B，crop=1。

obs1 右侧广告牌包含 GUS FIELD 2128415965 与 MARIA TRAVLOS 2127136795 两行；正确将后者与 B 绑定，一次裁图后停止。

类别：目标定位、姓名与号码绑定。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_grpo / sample 0**：refuted；gold=B，pred=A，crop=5。

首裁是道路；随后向左搜，五次裁图都未到右侧目标牌。Step3 称左侧裁图出现 MARIA，终答称 obs5 左下显示号码，均无该视图依据。选 A 的号码确实存在于右侧原图，但属于 GUS FIELD，不能据此认作目标号码。

类别：L1 搜索；L3 实例绑定；L4 所引视图不支持。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C04 · vstar_82

[结构化轨迹](C04.json)

**k8_dapo / sample 0**：supported_with_minor_error；gold=D，pred=D，crop=2。

首轮把罐位置描述为地面，obs1 后明确为长椅；obs2 的 56×56 原区域可见绿色罐，答案 D 有依据。嵌套框被最小裁图边长扩到56像素，未将该扩框当坐标幻觉。位置描述有小修正，不作为完整事实纠错正例。

类别：有效局部复看；轻微位置描述修正。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_grpo / sample 0**：refuted_process_correct_answer；gold=D，pred=D，crop=8。

obs1 已包含绿罐。其后有裁到楼房、路人、旗杆的动作；Step6 称罐在手里，后又称在长椅。最终引用 obs8 断言绿罐，但 obs8 左界225而罐约在原图x100，未包含罐。答案 D 正确，早期视图确有支持，但最终所引视图及若干中间对象陈述错误。不能因最终答对抹掉这些错误。

类别：L1 搜索偏离；L3 目标漂移；L4 视图引用；成本。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C05 · mme_10136

[结构化轨迹](C05.json)

**k8_dapo / sample 0**：refuted；gold=C，pred=B，crop=1。

obs1 清楚显示 Mar-26=10170、Mar-26 B=10251；问题问 budget。模型读取了 Actual 列10170，忽略 B 列限定，选 B。该例不支持 gold 错误的怀疑，gold C 与 Budget 列一致。

类别：L3 日期与预算列绑定。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_gspo / sample 0**：refuted；gold=C，pred=B，crop=1。

同样读到 Mar-26 实际值10170并选 B；裁图中预算列 Mar-26 B=10251 也可见。不是单纯没裁到目标值。

类别：L3 日期与预算列绑定。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C06 · worldbench_1160

[结构化轨迹](C06.json)

**k8_dapo / sample 0**：unsupported_process_correct_answer；gold=C，pred=C，crop=6。

六次裁图反复寻找尺子刻度，最终称 obs6 显示越过9米并计算约9.8米。复核原区域和 env.present 放大图，未见可支持9米或9.8米的标尺读数；obs6 也未覆盖完整鱿鱼。C 对上 gold，但读数依据未成立。不在本审查中独立确认真实物理长度。

类别：L4 无依据精确读数；L6 反复观察后无支持承诺。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_grpo / sample 0**：insufficient_evidence_correct_answer；gold=C，pred=C，crop=0。

零裁图直接称约9米，C 对上 gold；原图没有足以由本次审查复算出9米的明确尺度。不能把更便宜的答对直接列为证据充分的正例，也不能断言真实长度不是9米。

类别：终答正确与证据充分分离。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C07 · mme_14291

[结构化轨迹](C07.json)

**k8_dapo / sample 0**：refuted；gold=B，pred=A，crop=1。

obs1 读出的 one 是真的，但裁到了较左的灰色 ONE 旗，未确认题目所指靠大树左侧的黑旗。存在正确 OCR 绑定错误对象的现象。

类别：L1 候选选择；L3 对象绑定。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_grpo / sample 0**：supported；gold=B，pred=B，crop=2。

先取宽景保留树与旗位置，再对靠树的黑旗放大；obs2 清晰显示 BEAUTY STORE&SALONS，与 B 一致。两次观察都有可说明的定位／读取作用。

类别：上下文定位、嵌套精读。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C08 · worldbench_1378

[结构化轨迹](C08.json)

**k8_dapo / sample 0**：refuted；gold=A，pred=D，crop=1。

obs1 字母清晰：WORLD 在上、O F 在中、HYATT 在下；F 上方是 L，O 在 F 左侧同一行。模型读对字符串却把横向相邻当上下关系，选 D（O）而非 A（L）。

类别：L3 空间绑定。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C09 · vstar_95

[结构化轨迹](C09.json)

**k8_dapo / sample 0**：refuted_target；gold=A，pred=B，crop=1。

首轮把斜跨身体的包带当 watchband 候选，裁到后仍以蓝色包带作答。目标类别从腕表带变成包带，未在终答前校验。本轮只确认对象错误，不据此独立确认 gold 的紫色。

类别：L3 对象身份。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C10 · mme_3158

[结构化轨迹](C10.json)

**k8_dapo / sample 2**：refuted；gold=B，pred=E，crop=3。

obs2 可见路边两只独立轮胎。最终又把车轮计入，先说总四、再列2+2+2=6；REVISE 声称把路边四个改为两个，但此前并未断言四个。E 的含义是图中没有对象，不是数量不在选项中。即便计数单位有歧义，6→E 也不成立。

类别：L4 计数／修订不一致；L5 E语义。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

**k8_gigpo / sample 2**：refuted；gold=B，pred=D，crop=3。

obs1 为自行车，Step2 说自行车；终答却称 obs1 是停放汽车的两只轮胎。最终得6又选最接近4，题目未要求最近值。保留 tire 与 wheel 的标注单位疑问，不将它当作 gold 已证错误。

类别：L3 事实绑定漂移；L5 最近选项迁就。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C11 · validation_Geography_1

[结构化轨迹](C11.json)

**k8_dapo / sample 1**：refuted；gold=E，pred=F，crop=1。

首轮 commit 接 grounding，首轮即 HOLD/REVISE，记录了协议违规；该次评测仍执行一裁。多次因不匹配选项而改式；裁图切掉下方4.5与5的标注。终答算(-1,4)却说对应 F，而F为(1.5,-4)。不需要裁决扭矩正负约定即可确认选项映射与自洽性错误。REVISE 发生在新图之前，不是新视觉证据纠错实例。

类别：L0 协议；L1 裁掉标注；L4 推导一致性；L5 映射。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。

## C12 · mme_10450

[结构化轨迹](C12.json)

**k8_base_grpo / sample 0**：refuted_protocol；gold=D，pred=None，crop=0。

缺 think/action 标签；文本把折线数当Revenue，反复重复2020超过10000。8192生成token、finish=length，n_crops=0，未形成可执行工具轨迹或答案。此例说明协议及重复退化，不能解释为无SFT模型已经完成正常工具任务但视觉能力很弱。

类别：L0 协议；L3 系列绑定；重复／截断。视图不在公开包中；请按该条 source_image_sha256 与 boxes_abs 在本地重建。
