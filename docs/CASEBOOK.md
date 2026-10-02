# 五条真实轨迹：正例与未入池案例

本页来自已冻结记录。保留真实短 caption 与动作，没有补写成功的 REVISE。图片未随本公开仓库分发，因此本页只能检查文本、动作与时序，不能独立判定像素事实。完整图像核验曾在本地进行；公开数据中保留视图哈希、源尺寸、呈现尺寸和父子关系，便于合法取得图像后对照。

这些 `<caption>` 是 API 的可见协议文本，不是隐藏推理。导出时由协议适配为 `<think>` 短证据块；详情见 [协议](02_protocol.md)。

## KEEP：原图直接读取表格

`mme_realworld_lite_10768` / `k3-low` · [结构化记录](../cases/mme_realworld_lite_10768__k3-low.json)

原图足以读取所需行列时可直接提交；无需制造一次 crop。

问题：

```text
Which month shows the highest EBITDA, according to the 'Income Statement' table?
(A) May
(B) June
(C) July
(D) August
(E) The image does not feature the number.
```

实际答案：`C`；冻结金标：`C`。

人工记录：**KEEP**。EBITDA 行 Jun 489,248 / Jul 741,538 / Aug 653,785，与表一致；数字在 2548px 图上可读

|视图|父视图|源像素|呈现像素|原图归一化区域|
|---|---|---|---|---|
|original_image|—|[2560, 776]|[2548, 784]|[0.0, 0.0, 1.0, 1.0]|

### Step 1

这一步之前已收到：`original_image`。

```text
<caption>
Step 1 | commit
The Income Statement EBITDA row shows monthly values peaking at Jul Month 7 = 741,538, above Jun 489,248 and Aug 653,785 [original_image]. Among the options, July is highest.
</caption>
<answer>C</answer>
```

## KEEP：一次裁图补充局部证据

`worldbench_1844` / `glm` · [结构化记录](../cases/worldbench_1844__glm.json)

检查局部方块与公司标签的对应。不是裁得越多越好。

问题：

```text
Which company doesn't have a colored square next to the text box containing its name?
(A) Berkshire Hathaway Inc.
(B) Walmart Inc
(C) Exxon Mobil Corporation
(D) Microsoft Corporation
```

实际答案：`A`；冻结金标：`A`。

人工记录：**KEEP**。obs1 中 J&J 框左上有深蓝方块、Berkshire 框旁无方块，与 caption 一致；其余三个选项（Walmart、Exxon、Microsoft）在原图上确有方块。

|视图|父视图|源像素|呈现像素|原图归一化区域|
|---|---|---|---|---|
|original_image|—|[1600, 800]|[1596, 812]|[0.0, 0.0, 1.0, 1.0]|
|observation_1|original_image|[512, 272]|[840, 448]|[0.68, 0.24, 1.0, 0.58]|

### Step 1

这一步之前已收到：`original_image`。

```text
<caption>
Step 1 | locate
The four option labels sit in different areas: Microsoft inside a callout at mid-left, Berkshire Hathaway and the right-side label stack, and Walmart/Exxon at the bottom. I need to check each label for a small colored square marker. First I inspect the right-side stack where Berkshire Hathaway appears, since a square there is not clearly visible.
</caption>
<grounding>{"bbox_2d":[0.68,0.24,1.0,0.58],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_1`。该图不能作为本步已有事实的事后补证。

### Step 2

这一步之前已收到：`original_image, observation_1`。

```text
<caption>
Step 2 | commit
In [observation_1], the Johnson & Johnson label has a dark blue square marker at its upper left, while the Berkshire Hathaway Inc. (BRK.B), 1,2% box shows no colored square beside its text box. Berkshire Hathaway is option A, so it is the company lacking a square marker.
</caption>
<answer>A</answer>
```

## KEEP：跨视图保持两个小计

`mme_realworld_lite_22254` / `glm` · [结构化记录](../cases/mme_realworld_lite_22254__glm.json)

3 行 HOLD、2 个不同事实、1 条轨迹；不能计成 3 条覆盖。

问题：

```text
How many cars are running on the road in the top right corner of this picture?
(A) 3
(B) 5
(C) 6
(D) 9
(E) This image doesn't feature the count.
```

实际答案：`D`；冻结金标：`D`。

人工记录：**KEEP**。上行 6 辆（obs2、obs5 逐辆核对）、下行 3 辆（obs3）与 caption 一致，CALC 6+3=9 依赖两个被 HOLD 的小计；无坐标描述。

|视图|父视图|源像素|呈现像素|原图归一化区域|
|---|---|---|---|---|
|original_image|—|[7360, 4912]|[1708, 1148]|[0.0, 0.0, 1.0, 1.0]|
|observation_1|original_image|[1840, 982]|[1848, 980]|[0.75, 0.0, 1.0, 0.1999185667752443]|
|observation_2|observation_1|[626, 295]|[952, 448]|[0.8199728260869565, 0.0, 0.9050271739130434, 0.06005700325732899]|
|observation_3|observation_1|[828, 540]|[840, 532]|[0.8875, 0.06005700325732899, 1.0, 0.16999185667752442]|
|observation_4|observation_1|[1104, 639]|[1092, 644]|[0.85, 0.02992671009771987, 1.0, 0.16001628664495113]|
|observation_5|observation_1|[1104, 393]|[1260, 448]|[0.8125, 0.0, 0.9625, 0.08000814332247556]|

### Step 1

这一步之前已收到：`original_image`。

```text
<caption>
Step 1 | locate
The top right corner contains a road/bridge crossing near the water. I will crop that region to count the moving cars on it.
</caption>
<grounding>{"bbox_2d":[0.75,0.0,1.0,0.2],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_1`。该图不能作为本步已有事实的事后补证。

### Step 2

这一步之前已收到：`original_image, observation_1`。

```text
<caption>
Step 2 | re_examine
The corner crop shows two roads with vehicles: several on the upper curved road and a few on the lower right-hand road. I need to verify each vehicle's identity and count, starting with the upper road cluster near the top.
</caption>
<grounding>{"bbox_2d":[0.28,0.0,0.62,0.3],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_2`。该图不能作为本步已有事实的事后补证。

### Step 3

这一步之前已收到：`original_image, observation_1, observation_2`。

```text
<caption>
Step 3 | re_examine
The upper-road cluster in [observation_2] shows six vehicles moving in the same direction. The road continues rightward in [observation_1] where more vehicles (an orange car, a dark car, and a white van) appear; I need to check that area and the right edge for additional cars.
</caption>
<grounding>{"bbox_2d":[0.55,0.3,1.0,0.85],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_3`。该图不能作为本步已有事实的事后补证。

### Step 4

这一步之前已收到：`original_image, observation_1, observation_2, observation_3`。

```text
<caption>
Step 4 | re_examine
HOLD: [observation_2] the upper curved road in the corner shows six running cars (including the one cut by the top edge).
Observation_3 confirms three more vehicles on the lower carriageways: an orange car, a dark car, and a white van. I still need to check the band between and below them for any additional running cars before totaling.
</caption>
<grounding>{"bbox_2d":[0.4,0.15,1.0,0.8],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_4`。该图不能作为本步已有事实的事后补证。

### Step 5

这一步之前已收到：`original_image, observation_1, observation_2, observation_3, observation_4`。

```text
<caption>
Step 5 | re_examine
HOLD: [observation_3] the lower carriageways in the corner show exactly three running vehicles: an orange car, a dark car, and a white van.
The upper road's car cluster needs one unified count: [observation_2] and [observation_4] overlap there and their edge cars must not be double-counted. A wide crop along the whole upper road in [observation_1] will show every car on it in one frame.
</caption>
<grounding>{"bbox_2d":[0.25,0.0,0.85,0.4],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_5`。该图不能作为本步已有事实的事后补证。

### Step 6

这一步之前已收到：`original_image, observation_1, observation_2, observation_3, observation_4, observation_5`。

```text
<caption>
Step 6 | commit
HOLD: [observation_3] the lower carriageways of the corner bridge show exactly three running vehicles: an orange car, a dark car, and a white van.
The unified view [observation_5] shows the whole curved road with exactly six running cars (dark car at top edge, silver car, four dark cars in the cluster); [observation_4] confirms the road right of the cluster holds no more cars, so its top-left dark car is the same sixth one, not an extra.
CALC: 6 (upper curved road) + 3 (lower carriageways) = 9 cars running on the roads in the top right corner.
</caption>
<answer>D</answer>
```

## WEAK：有 HOLD 字符串仍未通过

`worldbench_1789` / `glm` · [结构化记录](../cases/worldbench_1789__glm.json)

人工记录指出旧值无声改变、新绑定伪装成旧事实；展示审计残余盲区，不作正例。

问题：

```text
Which of the following statements is true?
(A) Angle B2 is equal to 2 * angle A2
(B) Angle B1 is equal to Angle A1
(C) We cannot determine angle A1 + angle B1
(D) B1 to C has the same length as B1 to A1
```

实际答案：`C`；冻结金标：`C`。

人工记录：**WEAK**。几何读数与图一致且 A/B/D 逐项量化排除，但第 5 步 HOLD 里 'B1A1≈263 vs B1C≈386' 是此前没出现过的新绑定（此前量的是蓝杆 A2B2/B2C），θA2 在 45° 与 52° 之间无声改写，θA1 124/127–130 前后不一。

|视图|父视图|源像素|呈现像素|原图归一化区域|
|---|---|---|---|---|
|original_image|—|[809, 672]|[812, 672]|[0.0, 0.0, 1.0, 1.0]|
|observation_1|original_image|[161, 121]|[588, 448]|[0.5203955500618047, 0.7604166666666666, 0.7194066749072929, 0.9404761904761905]|
|observation_2|original_image|[202, 121]|[756, 448]|[0.6205191594561187, 0.41964285714285715, 0.8702101359703337, 0.5997023809523809]|
|observation_3|original_image|[194, 148]|[588, 448]|[0.28059332509270707, 0.6994047619047619, 0.5203955500618047, 0.9196428571428571]|
|observation_4|original_image|[178, 141]|[560, 448]|[0.10012360939431397, 0.44047619047619047, 0.32014833127317677, 0.6502976190476191]|
|observation_5|original_image|[162, 121]|[588, 448]|[0.10012360939431397, 0.41964285714285715, 0.3003708281829419, 0.5997023809523809]|

### Step 1

这一步之前已收到：`original_image`。

```text
<caption>
Step 1 | locate
Testing option (A) requires reading θA2 and θB2 from the drawn arcs, and option (D) needs the link lengths near B1. I first pin the A2 base center and its horizontal reference line.
</caption>
<grounding>{"bbox_2d":[0.52,0.76,0.72,0.94],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_1`。该图不能作为本步已有事实的事后补证。

### Step 2

这一步之前已收到：`original_image, observation_1`。

```text
<caption>
Step 2 | re_examine
The A2 view shows θA2 measured from a horizontal reference up to the blue link; the slope reads roughly 45° (dx≈dy over the link rise). I now need θB2, whose arc sits at joint B2 in the original view, to test option (A).
</caption>
<grounding>{"bbox_2d":[0.62,0.42,0.87,0.6],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_2`。该图不能作为本步已有事实的事后补证。

### Step 3

这一步之前已收到：`original_image, observation_1, observation_2`。

```text
<caption>
Step 3 | re_examine
At B2 the arc θB2 spans from the extension of link A2–B2 to link B2–C; measuring the crop, the extension sits ≈52–57° above horizontal and B2–C ≈127–131°, giving θB2 ≈ 70–79°, while 2×θA2 ≈ 2×52–54° ≈ 105–108°, so (A) looks false. The two blue links are also visibly unequal (A2B2 ≈ 258 px vs B2C ≈ 386 px). Now I must read θA1 at base A1 to test (B).
</caption>
<grounding>{"bbox_2d":[0.28,0.7,0.52,0.92],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_3`。该图不能作为本步已有事实的事后补证。

### Step 4

这一步之前已收到：`original_image, observation_1, observation_2, observation_3`。

```text
<caption>
Step 4 | re_examine
θA1 reads ≈127–130°: the arc runs from the horizontal reference counterclockwise to the up-left link (slope ≈ dy/dx giving ~127°). Now the unresolved binding is θB1's arc extent at joint B1, needed to test option (B).
</caption>
<grounding>{"bbox_2d":[0.1,0.44,0.32,0.65],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_4`。该图不能作为本步已有事实的事后补证。

### Step 5

这一步之前已收到：`original_image, observation_1, observation_2, observation_3, observation_4`。

```text
<caption>
Step 5 | re_examine
The θB1 arc sweeps the long way around (left–bottom–right) from the extension of A1–B1 (~126°) to B1–C (~49°), so θB1 reads as a reflex ≈282° (minor arc ≈78°) — clearly not equal to θA1 ≈ 126° [observation_3, observation_4].
HOLD: [observation_1] θA2 ≈ 52°; [observation_2] θB2 ≈ 79°; [original_image] B1A1 ≈ 263 px vs B1C ≈ 386 px.
</caption>
<grounding>{"bbox_2d":[0.1,0.42,0.3,0.6],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_5`。该图不能作为本步已有事实的事后补证。

### Step 6

这一步之前已收到：`original_image, observation_1, observation_2, observation_3, observation_4, observation_5`。

```text
<caption>
Step 6 | commit
Read values: θA2 ≈ 51° [observation_1], θB2 arc spans from the link extension (≈53°) to B2–C (≈131°) [observation_2], θA1 ≈ 124° [observation_3], θB1 is drawn as the reflex sweep from the A1–B1 extension to B1–C [observation_4, observation_5].
RULE: angles measured CCW from the horizontal reference as drawn; link length = pixel distance between joint centers.
CALC: 2×θA2 ≈ 102° vs θB2 ≈ 131° − 53° ≈ 78° → (A) false.
CALC: θB1 ≈ 360° − (124° − 51°) ≈ 287° ≠ θA1 ≈ 124° → (B) false.
CALC: |B1–C| ≈ √(252² + 309²) ≈ 399 px vs |B1–A1| ≈ √(151² + 218²) ≈ 265 px [original_image] → (D) false.
The drawing gives no numeric angle data, so θA1 + θB1 cannot be determined exactly; (C) is the true statement.
</caption>
<answer>C</answer>
```

## 未入池：真实 fallback，终答不符金标

`mme_realworld_lite_4726` / `sonnet` · [结构化记录](../cases/mme_realworld_lite_4726__sonnet.json)

裁到树和车后换区域，确实取得了人物视图；终答 A 与冻结 gold C 不符，且 caption 有坐标。不可据此说完全没有发生回退。

问题：

```text
This image shows the front view of the ego car. What should the ego vehicle do when encountering the pedestrian wearing blue top on the left?
(A) passing by
(B) slowing down
(C) no response
(D) stopping
(E) The image does not feature the object
```

实际答案：`A`；冻结金标：`C`。

|视图|父视图|源像素|呈现像素|原图归一化区域|
|---|---|---|---|---|
|original_image|—|[5760, 1200]|[3080, 644]|[0.0, 0.0, 1.0, 1.0]|
|observation_1|original_image|[2880, 600]|[2884, 588]|[0.0, 0.3, 0.5, 0.8]|
|observation_2|observation_1|[259, 162]|[728, 448]|[0.25503472222222223, 0.425, 0.3, 0.56]|
|observation_3|observation_1|[230, 138]|[756, 448]|[0.45, 0.425, 0.48993055555555554, 0.54]|

### Step 1

这一步之前已收到：`original_image`。

```text
<caption>
Step 1 | locate
No pedestrian in a blue top is visible in the overview [original_image]. The left side shows parked cars, a tree, and a sidewalk, so I will crop the left half to check for a small person.
</caption>
<grounding>{"bbox_2d":[0,0.3,0.5,0.8],"source":"original_image"}</grounding>
```

动作执行后收到：`observation_1`。该图不能作为本步已有事实的事后补证。

### Step 2

这一步之前已收到：`original_image, observation_1`。

```text
<caption>
Step 2 | locate
A small figure with a blue top is visible near the right edge of [observation_1], beside the distant parked cars on the left side of the road. I will crop it to check its position and posture.
</caption>
<grounding>{"bbox_2d":[0.51,0.25,0.6,0.52],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_2`。该图不能作为本步已有事实的事后补证。

### Step 3

这一步之前已收到：`original_image, observation_1, observation_2`。

```text
<caption>
Step 3 | fallback
The crop in [observation_2] landed on the dark tree and the black car and shows no person. In [observation_1] the blue-topped figure is near the right edge, beside the far parked cars, at about x=0.93 and y=0.36. I will crop that region.
</caption>
<grounding>{"bbox_2d":[0.9,0.25,0.98,0.48],"source":"observation_1"}</grounding>
```

动作执行后收到：`observation_3`。该图不能作为本步已有事实的事后补证。

### Step 4

这一步之前已收到：`original_image, observation_1, observation_2, observation_3`。

```text
<caption>
Step 4 | commit
[observation_3] shows a pedestrian in a light-blue top carrying a green bag, standing on the left side of the road between parked cars at the curb. This is the left-side blue-topped pedestrian. The person is far ahead and at the curb, not in the ego lane, so the ego car can keep going and pass by.
</caption>
<answer>A</answer>
```

