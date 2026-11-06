# PathGenModel 研究交接

## 2026-11-06：编辑器保存PNG/存档总览、刚体旋转、指定参照线

用户准备手绘5–10原型，要求保存图片、旋转线/box、测距不被近路沿干扰。已完成scene_editor.py及tools/scene_editor改动：
- 每次保存新增scene.png，完整编辑几何自动取景、名称/nav/ID/米制坐标/ego，粗Guide标前80m；返回preview_path/url，界面可点图。PNG在创建输出目录前渲染，短Guide仍先拒绝。Matplotlib Figure/Agg+锁避免HTTP线程绘图竞争。
- outputs/scene_editor/gallery.html自动更新离线缩略图总览，链接大图/scene/input，仅浏览不删除。--backfill-previews只补缺失PNG、不改原JSON或覆盖旧图，已为原16个存档补图，无错误。浏览器测试另产生标名browser_validation的新存档。
- 新geometry.js：旋转线/Bezier所有控制点和区域顶点，支持数字角度及±5°，中心/首点/末点/ego支点。正角左旋，Guide默认首点。中心为按几何线长加权的中心（框含闭合边），刚体变换下保持稳定，避免连续旋转漂移。撤销/重做有效。
- 新参照线下拉框：自动最近或指定ID；按现有类型筛选候选，高亮指定参照，拖动/旋转时保持锁定；删除/滤掉参照则回退自动。
- GET /api/health editor_version=3；旧后台提示重启。GET /api/scene-image/<时间戳>.png严格路径白名单。
服务原PID796360已确认后停止，现同地址127.0.0.1:8765后台PID2741837，日志outputs/scene_editor/server_20261106_154533_619634/server.log；旧浏览器草稿保留，用户先下载草稿备份再刷新加载新UI。
13项tests/test_scene_editor.py通过（新增PNG非空/格式/画廊及补图不覆写），Node geometry/measurement测试通过（旋转方向/支点/逆变换/形状/Bezier/指定远参照），Chrome真实浏览器操作全部通过，含曲线/框旋转与撤销、近路沿.2m时锁定另一线、PNG HTTP读取、保存/导入/离线重连。
最终截图outputs/scene_editor/browser_validation_IW6rUy/editor_features.png；说明README_SCENE_EDITOR.md已更新。未改用户examples/custom_input_split.json、训练或模型。

## 2026-11-06：用户重训混合增强模型已完整验证

权重已恢复到outputs/training_road_mixed_guide_aug_v1/best.pt，best90/last115，训练best.2387576，25轮早停。SHA256 ab1965af887610c8bba23afa631813682d83f14476db1f8d139b5eee0190e23e。用户明确授权执行验证，本次CPU threads1/num_workers0，未训练/改模型或数据。
常规评估新目录eval_val_20261106，6291/90World，输入指纹一致、train/val World无交集：ADE.23874949、FDE.33921924、EmptyBEV.55200291；路口1890 ADE.61610390、Empty1.46389094；分流4401 ADE.07669545、Empty.16039456。与旧增强结果基本持平。
汇总outputs/model_review/20261106_mixed_guide_aug_v1/findings.md和summary.json。standardized重跑新混合+旧单路口：mixed/单路口ADE：straight .5925/.6332、left .2999/1.3265、split .5103/.5934；mixed出口横向绝对均值 .7511/.1982/.7044m。均无完整道路/目标走廊越界或路沿相交（仅中心线）。标准左转清空BEV ADE .2999→4.4591，证实该例依赖道路；直行分流居中仍不足。三例人工标签非真实GT，几何/FOV/语义同步标准化非因果隔离。
custom_left重跑原183404、183817，最新mixed仍首例偏相邻出口、第二例穿curb1(x16.634,y17.470)，与误删前保存预测平均差.106/.132m，旧问题大体重现；无GT不称ADE。新overlay各在standardized/comparison.png、custom_left/comparison.png。
结果不足以支持“混合模型全面差于单路口”或“完全忽略BEV”，也未解决自定义泛化。继续保留当前baseline，感知概率性平顺形变的代码尚未改动，后续先小预览再训练。

## 2026-11-06：用户审核感知预览，要求平顺的结构性拟合偏差

原话评价已存outputs/perception_review/20261105_v1/user_feedback_20261106.json。A路口：clean A/B太理想，中A优于B、强A优于B。B路口：clean A/B太理想，中B优于A、强B优于A。C/D增道：clean A/B及medium A/B均太理想，hard A优于B。相对偏好不等于未选样例绝不合理，不能由此直接标定发生概率。
用户新示意图：左道路/参考，右紫色感知是平顺线条，不同起止/可见长度，部分线局部弯曲/倾斜/曲率偏差，右侧线有缺口；想要有概率出现此类误差，并非每帧硬造。当前observe的片段平移+点抖动不足，应优先考虑低频连续几何/拟合偏差及可见段变化，保留正常/较准观测与不同线独立失效的混合。只保存反馈并解释图意，尚未改感知器或训练。

## 2026-11-05：已生成4场景28份感知观测供用户定性审核

用户同意先生成，不等当前重训完成。新增scripts/build_perception_review.py及README_PERCEPTION_REVIEW.md；CPU生成outputs/perception_review/20261105_v1，未训练/推理，不改既有observe/训练数据。
四场景：A路口左转seed30000000、B路口右转30007919，均entry前5m；C右侧2→3增道44000000、D左侧3→4增道44000004，均过渡前8m；仅增道无减道，保留物理unmarked公共空白。
每例同World/ego/GT/Guide/nav，主对照FOV统一85前/18后/±22侧，基准无几何噪声/随机缺失+当前clean/medium/hard误差参数各A/B两次固定seed直接抽样，不筛选。图另有物理真值面板；共28输入，实际模型编码最多32×32，全部无容量丢弃。range_comparison.png另单独示当前三档FOV差异且关随机扰动。
入口gallery.html可给每张标常见/偶尔有/不合理+备注，本地存储、下载feedback JSON，不上传。各例comparison.png、detail.png、各档独立PNG/输入JSON、world/truth_ego/context，manifest记录参数/seed/几何偏差统计。浅灰物理真值背景不是输入，浅黄=物理无内部标线区，绿GT橙Guide非模型预测。
已查看A/C整组图、验证28反馈卡片及所有链接存在、JS语法、同场景GT/Guide/FOV不变量、增道约束、编码有限数值、2例按manifest种子逐对象精确重现。噪声裁剪后可能少量越FOV边，保留现有模拟器行为；角度噪声主要用于区域框，线条为平移+点噪声且每可见段最多一缺口。等待用户定性评价再调参数；不宣称已真实标定。

## 2026-11-05：用户收窄分流范围为增道；感知无需用户先定量

用户明确纠正：下一版分流暂不考虑减道，只考虑增道；这是后续生成范围，不改既有训练数据或打断其当前重训。用户确认误删最新混合权重且无备份，已自行重新训练。
重新核实当前600World分流generation_report及train样本、build_samples：正式数据45360行，调用observe，clean/medium/hard均已叠加FOV裁剪/漏检/可见段断裂/位置噪声和偏移，convention vcs_visible_segment_breaks_v2；混合分流子集也保留该约定。物理World600例全部unmarked，物理空白区不等同感知漏检。前一轮新标准化3例只做确定性裁剪，无随机感知噪声，不应与正式训练数据混淆。
用户难以定量描述目标感知；后续应基于其画的可见片段原型做同World分档预览，用定性审核收敛，不要求先给漏检概率/噪声方差。无真实统计时现有参数只是可解释假设，不声称真实标定；先查清洁/轻扰动下道路修正，再增加困难观测。

## 2026-11-05：标准化三例已运行单路口权重；当前缺失最新混合增强权重

用户问标准化结果，发现此前未有预测，遂执行CPU三例。当前工作区outputs/training_road_mixed_guide_aug_v1整个目录不存在，原mixed_v2也不在；已查本项目及/mnt可见候选，未找到目标权重；异步问用户新混合权重位置，尚待回复。不用context_comparison_inside13/mixed替代最新模型。
单路口guide_training_v1/aug/best.pt仍在，结果outputs/custom_comparison/20261105_standardized_v2，comparison.png、summary.json、findings.md、compare.py，三子例intersection_aug含input/prediction.json/png。
人工参考ADE Guide→模型：straight .6801→.6332、left_turn2.6231→1.3265、right_split .8350→.5934m。出口横向绝对均值 .8→.7774、2.5→1.2425、.8→.5250m（投影参考，出口按参考s>=40/38/60的同索引预测点）。三个中心线均无完整道路/目标走廊外长度、无观测或完整World curb相交；左转向内Guide偏差被修到向外1.24m，仍过修。非车辆轮廓安全检查。
当前只评单路口，混合对照未完成，不能把此结果称为混合模型或因果证明原自定义不真实。需要用户提供/同步最新混合权重后用新输出目录补对照。compare.py ROOT=parents[3]，可复制到同层新目录复跑避免旧结果覆盖。

## 2026-11-05：标准化、目标已标注的自定义诊断样例

用户要求创建连接明确、FOV统一、目标已标注版本。新增独立脚本scripts/build_standardized_custom_cases.py，输出outputs/custom_standardized/20261105_v2（v1为本轮初始预览保留），3例straight/left_turn/right_split；gallery.html、overview.png、各例preview/input/scene/annotation/world、validation.json。说明README_STANDARDIZED_CUSTOM_CASES.md。
按原意图重建，非原图自动GT，也非完整四臂路口复刻。车道3.5m，曲线切向连续，自车/Guide/参考起点0，Guide和参考各81点。左转为双车道90°连接，nav当前left，Guide提前2.5m并出口向内偏2.5m、有意保留curb冲突。分流为明确2→3自然扇出目标新增右出口，nav为后续右转准备，过渡x12–52、空白16–46，Guide滞后8m且偏左.8。直行navstraight/Guide渐偏右.8。
BEV严格[-18,85]x[-22,22]米裁剪，不加随机噪声/漏检，.99置信度；规划Guide/参考不裁FOV。world/targetcorridor侧文件仅审计，input带gt_path仅人工参考；编码器带/不带gt逐张量相同。自动检查道路/走廊覆盖、无token丢弃、FOV、采样、Guide起点、编辑器roundtrip通过。scene保留完整密集Guide，避免81点弦长<80导致重新保存被拒。
已查看左右预览；图中绿色/浅绿都是人工标签非模型预测。未运行模型/训练，也未改既有生成器或用户数据。标签不作为额外模型输入，不声称外部标注自动解决可观测性问题。

## 2026-11-05：执行单路口Guide增强权重复跑左转，未支持混合是主要原因

用户明确要求代跑此前几个真正左转案例。使用guide_training_v1/aug/best.pt CPU跑183404和183817两份输入；134852与183404完全相同不重复。结果新存outputs/custom_comparison/20261105_intersection_guide_only/case_183404和case_183817，overlay comparison.png、summary.json、compare.py、findings.md。
逐份输入完全一致。第一例单路口和混合增强都避给定curb，但均偏至相邻车道；y40截面相对line1/line2中点偏差混合4.652、单路口4.968m。第二例均穿curb1（混合x16.570/y17.611，单路口x15.773/y19.930）；单路口偏差稍小仍失败。非GT不称ADE。
已向用户修正：近直道表现更好不能推广到左转，本次不支持把主要问题归因混合负迁移。共享数据规则/道路-Guide覆盖与几何约束仍是待查因素，混合干扰不排除。未修改输入或权重，未训练。

## 2026-11-05：同一近直道输入，单路口增强模型优于混合模型的居中参考

custom_inference/20261105_185446_664301为mixed-guide-aug，185508_642331为单路口guide_training_v1/aug；input.json完全相同，模型均v1XY。Guide起点已改到(-.0782,-.2289)，预测首段.934/.978m，之前10m缺口已消除。
两预测逐索引平均差1.30479m、末点差1.11775m（非ADE）。若line1/line2为目标道路带，x40/60/75截面混合预测偏中点右1.022/.935/.812m，单路口偏左.572/.355/.329m。两者均未穿给定标线/路沿，无GT不可称ADE或全程安全。两者都相对Guide向左修正，混合幅度不足，单路口更接近中点。
支持该案例单路口模型居中更好、混合可能有任务干扰；不能从此单例证明整体负迁移或路口模型更依赖BEV，两个训练还存在感知重导出/预算差异。未改模型/输入。

## 2026-11-05：确认单路口Guide增强权重仍在；新直道案例起点前移

用户想比较custom_inference/20261105_185048_579894。已加载核实outputs/guide_training_v1/aug/best.pt（epoch78，val ADE.624594，train=v4_world300_inside13，Guide增强p.5）仍在且是v1 XY，同当前混合架构适合比较。v2 aug也在（epoch88，ADE.649179，output_frame guide_sl），不优先用于单因素对比。
新案例当前加载training_road_mixed_guide_aug_v1，navleft但Guide/道路基本直行。Guide起点(10.2619,.2866)，终点x90.254；预测强制p0原点，p1=(11.2528,.1501)，首段约11.25m，非正常1m路径步长。不能把起点非零一般性要求撤销，但本例10m切向缺口需单独标出，较此前小侧偏更严重。
x40 line1/line2界y2.535/-.885，Guide y-.141，预测-.759；目标若为该道路带则预测更贴右界。无给定标线/curb相交，无GT。给用户命令用归档input.json跑单路口v1权重，未代为推理或改输入。

## 2026-11-05：准备路口Guide增强 + 原分流混合训练

用户要求先尝试此前同路多Guide增强+分流，由用户训练，暂不改道路几何生成器。
新增configs/train_road_mixed_guide_aug_v1.json，沿用road_mixed_v2划分/结构/Loss/seed42/batch16/150epoch/25早停，输出outputs/training_road_mixed_guide_aug_v1，强制cuda；Guide cache outputs/guide_train_cache_road_mixed_v1，p=.5。
旧cache不能直接用于混合：脚本scripts/build_training_guide_cache.py新增--mixed-source outputs/road_mixed_v2；--source仍outputs/v4_world300_inside13提供物理World/routes。生成version2缓存，绑定完整混合训练指纹，仅选择intersection左右转；8变体/route、跨pose一致。Dataset/guide_augmentation.py兼容v1/v2缓存，分流（含左右导航）保持原Guide，路口直行保持原样；训练样本数不变，总增强率约9.8%。
README_MIXED_GUIDE_AUG.md给出CUDA检查、缓存构建、从头训练、评估命令。未启动完整缓存或正式训练。旧模型/数据/输出保留。
小型混合测试证明仅路口turn Guide改变，其余tensor逐项完全一致、训练指纹及缺失key拒绝；原确定性与resume测试通过。

## 2026-11-05：生成器与自定义场景的覆盖差异复核

重新核对Scenario/build_dataset_v4.py、build_lane_transition.py、World/intersection_generator*.py及当前混合配置。
共同：Guide/GT共享入口出口及道路变形参数，每route单Guide；联合v2 guide_augmentation=null。物理几何、Guide、GT相关性过强有学习模板修正规律风险，但不能据此认定泄漏或证明模型结构无问题。自车pose均位于GT并沿GT切向，缺横向/航向偏差恢复。
路口：base cfg w3.5/r9，四臂+共享ArmWarp（有随机角度/侧移及车道数，并非完全固定十字路口）；GT固定.45 chord控制柄，Guide仅入口±.4/出口±3/同一柄比例变化；缺独立起终点方向/局部曲率。nav决定出口臂，出入口lane为随机可行组合，Guide偏差可能令目标车道身份不明确，需目标/可达性审计。
分流：五模板、World等宽3.3–3.8、35–50m过渡、同一smooth基线、分段直线Guide、GT五次式；槽位差<=1误排2→4最外自然扇出。现有物理空白段/感知漏检断裂均已存在，问题非“无缺失”或“无冲突”。
新确认左转自定义134852出口线延伸至ego y≈71m；当前感知clean最大left22（medium15/hard9），所有三条出口长线大部在训练FOV外。直接在编辑器画远线没有与训练统一感知裁剪；这是接口分布不一致，非证明唯一原因，需决定部署FOV并做一致对照。
验证按World隔离正确，但仍同规则，且混合指标被约70%简单分流主导；需规则外挑战集、出口身份/边界/连续性指标。无改代码或训练。

## 2026-11-05：自定义路口趋势也不能证明道路理解

复核custom_inference/20261105_134852_861921，navleft，联合v2 best，全部6元素保留，无GT；5例CPU对照存outputs/custom_input_diagnostics/20261105_intersection_review。
Guide穿两段curb，预测均避开；但若目标为Guide所指line1/line2道路带，预测出口先走line2/line3带，再在(x20.94,y59.69)穿line2回切。y30截面相对line1/2中点Guide偏.850m，预测偏2.777m（截面参考非GT/ADE）；起步先右偏.671m再左转存在折点。
原预测相对Guide平均XY修正4.529m（含纵向），清空全部BEV后原预测平均只变.278m且仍避开原路沿、仍同类跨线。仅去curb.249m、去cross.090m、nav改straight.534m。不能把避沿直接归因于路沿输入，也不能说完全忽略BEV。
当前证据更符合Guide/nav及训练先验主导总体修正、实际边界控制和出口保持不足；单例不能推广为所有路口失败。下一步需共同挑战集与明确出口GT，做道路/Guide反事实对照。

## 2026-11-05：车道宽度调整后仍偏Guide（两个新案例）

复核custom_inference/20261105_132723_406979、133929_954093，并执行14例CPU固定checkpoint消融。报告outputs/custom_input_diagnostics/20261105_width_review/findings.md、summary.json、comparison.png。
两例Guide逐点相同且nav=straight，BEV改变预测平均只变5.22cm。清空BEV平均变化11.14/7.12cm，统一正/反点序均不影响，黄改白1.75/.42cm。全部元素保留、CPU复现误差<.1mm。
x70标线间距第一例3.80/3.50/2.89，第二例2.85/3.29/2.92m（横截面近似宽，部分仍比训练3.3–3.8窄）。若line3/line2道路带为目标，Guide→预测相对中点偏左1.123→1.440m、.180→.419m；第二例x30 .483→.744m。第一例x61.244穿过虚线line3，不能只凭虚线相交称违法；两例无给定路沿相交，无GT不称ADE。
说明宽度不是充分解释，更符合Guide先验/任务覆盖不足，但非严格宽度单因素训练或模型结构排除。建议先固定挑战集明确目标，再做同路多Guide（正负偏移/曲率）训练对照及出口稳定检查；未改生成器/模型。

## 2026-11-05：编辑器邻线测距与保存断线修复

用户反馈拖动难估车道宽、保存Failed to fetch。主机检查127.0.0.1:8765连接拒绝，原前台服务已退出，并非Guide长度/JSON编码报错。
新增tools/scene_editor/measurement.js，选中线的局部曲线测点→邻线分段投影最短距离，拖动/悬停实时显示辅助线、米值、邻线ID及测点坐标。默认仅BEV line/curb，可选仅line或含Guide；不是全线最短距离或法向道路宽度。Bezier控制柄先投影到实际曲线，不拿柄坐标当道路点。
新增/api/health、前端连接状态、断线明确恢复说明、失败立即保存localStorage及pagehide保留；下载草稿仍纯前端。
scene_editor.py --background [--open]支持独立终端运行，启动健康检查，日志/PID写新server_<时间>目录；不替换占用端口的旧服务。
已恢复8765后台服务，PID796360，日志outputs/scene_editor/server_20261105_132310_738153/server.log。旧页面可直接再保存，加载测距功能需先下载草稿备份再刷新。
12项Python导出测试通过；Node几何测试通过；Chrome浏览器验证真实拖动距离层、保存、模拟断线/草稿保留/重连保存及原有功能通过。截图outputs/scene_editor/browser_validation_brG2gC/editor.png，验证输入20261105_132326_748955/input.json。
未改模型、生成器或用户examples/custom_input_split.json。

## 2026-11-05：三个自由绘制分流案例与输入消融

用户反馈custom_inference/20261105_113815_926244、130333_555172、130508_350433效果一般。
复核原预测+15例CPU单因素消融，报告outputs/custom_input_diagnostics/20261105_three_scenes/findings.md，图three_scenes_ablation.png。
导出/语义/容量正常；CPU复现GPU最大差0.000363m。BEV清空预测平均变0.356/0.404/0.423m，改nav直行变0.675/0.315/0.311m，confidence1→.99只变0.2–0.3mm。第三例只去路沿变1.53cm，路沿与外侧标线冗余，不能断言完全忽略路沿。
第二、三例Guide一致、BEV多处不同，预测平均差2.75cm；非单独加路沿的严格对照。
若最右可见道路带是目标，x70处三预测偏离两界中点向左1.64/1.93/1.53m；Guide已经偏左，预测仍继续左移约0.4m。无GT，不能称ADE；宽间隔也可能是中间标线漏检。
主要疑点：训练五模板、等宽车道、共享边界变形和Guide/GT生成规律，覆盖自由曲率/非齐次标线/导航连接不够；此前槽位差约束过强仍未修订。验证7cm只表示同生成规则下World泛化，不代表自由场景已学会道路修正。未改模型/数据或训练。

## 2026-11-05：本地BEV场景编辑器已完成

用户要求先做工具：Guide/BEV line/curb，指定起终点，自由调整形状，选中编辑语义，选择导航后自动采样保存。
入口scene_editor.py（Python标准库HTTP，127.0.0.1:8765）；前端tools/scene_editor/，说明README_SCENE_EDITOR.md。导出使用已有infer.prepare_sample/encode_input，不改模型/训练。
支持两点创建三次Bezier并拖控制柄、折线增删节点、数字坐标、画布平移缩放、撤销重做、导入原JSON/编辑工程、置信度及车道线颜色/实虚线、导航选择。额外支持停止线/斑马线矩形。
保存到outputs/scene_editor/<时间>/input.json（Guide81点/前80m）、scene.json（控制点原稿）、model_input.npz（实际32×32×6编码及10维属性）。Guide短于80m报错，不拉伸，Guide起点偏移保留。
12项tests/test_scene_editor.py通过；Chrome真实浏览器测试通过拖动/绘制/属性/导航/撤销/保存/导入/短Guide错误。可复跑tests/scene_editor_browser_smoke.cjs（运行编辑器后node执行）。
已用导出JSON运行联合v2 best的CPU单样本infer，输出outputs/scene_editor/browser_validation/inference；浏览器截图outputs/scene_editor/browser_validation/editor.png。
已启动本机8765服务（本轮tool会话67197）；后续如服务已退出，用户运行python scene_editor.py --open。未修改examples/custom_input_split.json的用户改动。

## 2026-11-05：自定义分流仍偏Guide，发现目标覆盖缺口

复核custom_inference/20261105_095353_374408，联合v2、cuda。无GT；预测后40m相对Guide位移仅0.1058m，x79末段接近Guide且相对推断右侧目标车道中心偏右约0.972m。
给定路沿只到x65，预测中心线未交该已观测片段，不能作全程车身碰撞保证。全部7个元素保留。
重要生成规则不足：自定义近似2→4右侧增道，Guide指向最外新增出口3；训练按abs(exit-entry)<=1筛选，此模板入口0/1只见出口0/1/2，从未以出口3作GT目标。已核查urban_transition_0044000002/routes.json。
新增车道的自然扇出不等价于跨越多条已有车道，当前索引差约束过强；这是覆盖缺口，不是已证明的全部失败原因。自定义约79m外扩也超出当前35–50m训练长度。
分析存 outputs/custom_input_diagnostics/20261105_split_review/findings.md。用户询问绘图导出JSON方法，仓库暂无编辑器，建议本地BEV矢量画布导入/编辑/导出infer兼容JSON，尚未实现。

## 2026-11-05：新版联合训练完成并评估

training_road_mixed_v2 best为epoch101，第126轮早停，6291条验证ADE0.2349355m，FDE0.3361319m。
路口1890条ADE0.6122467m/EmptyBEV1.1904723m；新分流4401条0.0729000m/EmptyBEV0.1648056m/Guide0.6044817m。
分流clean/medium/hard ADE0.06908/0.07023/0.07939m；左右直导航均约6–8cm。
额外只分析保存预测：analysis_20261105/transition_region_summary.json与findings.md。
分流before10/during按GT世界x落入过渡段的局部ADE0.07220/0.07498m，公共内部无标线区0.07800/0.06569m（各1467例），EmptyBEV公共空白区0.28975/0.13919m。after5无可评价未来过渡点。
支持当前分布内分流修正较准且利用感知，不是仅靠出口长直线拉低ADE；仍未评价道路/车道合法性及自定义分流泛化。
路口entry1.10825m、approach10 0.95723m，仍是主要难点；约70%的分流验证样本使混合指标更低，不代表路口0.235m。

## 2026-11-04：联合训练实际在CPU，定位CUDA故障

用户发现epoch约530s。核实 training_road_mixed_v2/last.pt epoch59的cuda_rng=None，训练配置device=auto，实际退回CPU。
在主机（隔离环境外）复现 torch2.9.1+cu128 CUDA初始化失败；libcuda cuInit=999 CUDA_ERROR_UNKNOWN。
nvidia-smi能识别RTX3050 Laptop，设备节点权限/主设备号正常，未设置CUDA_VISIBLE_DEVICES或LD_LIBRARY_PATH/LD_PRELOAD。
内核日志：10月30日22:13:45休眠恢复，22:13:48 Xid31 MMU Fault，随后Xid154明确记录Node Reboot Required；本次启动始于10月21日，故障后未重启。
建议用户先停止训练，保留last/best，重启电脑后检查CUDA可用并用--resume last.pt --device cuda继续；未执行停止、重启或驱动改动。

## 2026-11-04：下一轮联合训练指令与配置已准备

用户要求自行执行混合/训练，路口选之前含Guide越界案例的 outputs/v4_world300_inside13。
新增 configs/train_road_mixed_v2.json 和 README_MIXED_ROADS_V2.md，混合输出 outputs/road_mixed_v2，训练 outputs/training_road_mixed_v2。
模型仍v1 XY residual+原Loss、seed42，从头训练，不开旧Guide缓存增强或FOV补充。最多150轮、25轮早停，按混合val ADE选best，分组记录两场景。
混合命令refresh-perception intersection，使用现有World统一路口感知断裂规则，GT/Guide/pose/split不变；分流已含感知，不重叠加。
预计train51462（路口15120+分流36342）、val6291、test6507，自然训练比例29.4%路口/70.6%分流。
本机尚未执行此新混合或正式训练；由用户运行文档命令。

## 2026-11-04：600 World正式批量与感知样本生成完成

用户授权批量生成，并强调World不能直接训练、必须叠加感知。已完成 outputs/urban_lane_transition_world600_v1，validation_report.json passed=true。
600 World，seed44000000，已确认的5模板各120 World；35–50m随机过渡，全部unmarked，分段一次Guide、left/straight/right导航。
每World已调用observe叠加clean/medium/hard的FOV裁剪、漏检、可见段断裂、位置噪声；训练输入不是物理World线。
合计45360行；train480 World/36342行，val60/4401，test60/4617。每种导航15120行，每档质量15120行，每个pose15120行。
28条hard空BEV样本按现有漏检设定保留；clean/medium无空BEV。所有GT/Guide形状、间距、质量间一致性及World划分校验通过，抽查454条实际编码。
训练应读取该目录audit/train.jsonl、val.jsonl；worlds保留物理真值用于审计和绘图。尚未混合路口或启动训练。
为控制内存，生成器改成逐World写盘；改造后5-World样本及三个划分与原版逐字节一致。正式批次4个150-World分片并行，最终按全局seed重新划分（不使用分片局部划分）。
串行中途输出保留在 outputs/urban_lane_transition_world600_v1_partial_serial，标记INCOMPLETE，不能训练；正式数据以无partial后缀目录为准。
可复核 generation_report.json、generation_provenance.json、validation_report.json；图集gallery.html及overview.png。

## 2026-11-04：用户确定过渡区35–50m随机

urban道路过渡长度由35–80m改为35–50m均匀随机；CLI默认marking-pattern改为unmarked。
无标线区名义起点为过渡长度10–20%、终点76–88%，各线端点另有±3.5%长度扰动，保留共同空白区。
新5 World预览 outputs/urban_lane_transition_preview_v4_length35_50，旧版保留，未训练。

## 2026-11-03 更新：补充物理内部无标线区域

用户指出上一版内部线过于完整。新增 --marking-pattern mixed|unmarked|partial|continuous，CLI默认mixed。
新预览 outputs/urban_lane_transition_preview_v3_marking_gaps：5 World，按无内部标线/部分缺失/连续/无内部标线/部分缺失排列。
原路沿、外侧实线、GT/Guide和导航目标保留，同seed不同标线模式只修改内部标线；感知随机序列可能变化。
完整缺失模式各线起止略不同，但保证共同空白区；两侧片段独立对象，编码不跨缺口连接。浅黄色标出公共空白区。
上一版v2及所有旧输出保留；未启动训练。

## 2026-11-03：城区增减道导航预览，尚未训练

用户确认要正常城区增减道 + 后续导航意图 + Guide 车道指引，避免跨多车道不合理跳转。
Scenario/build_lane_transition.py 新增 urban CLI 默认模式；旧 make_world 及 --scene-mode legacy 保留。
5 World预览 outputs/urban_lane_transition_preview_v2，378行、多导航/pose/quality；按World分3/1/1，只用于确认形态。
导航 left/right 选择可达范围物理左/右出口，straight 最小横移；最多相邻一槽位调整，不可达入口省略。
道路过渡35–80m；持续分隔线保留，新增/消失线在各自位置起止，Guide带位置和相位误差。
每张预览 nav行×pose列，clean质量，同一入口便于对比。尚未实现同路多Guide、车速/动力学约束。
本机没有 outputs/training_road_mixed_v1 及其数据；此前联合成绩来自交接记录，未复现联合模型自定义失败。
自定义输入 examples/custom_input_split.json 的 navigation=right，旧增减道全部straight；输入约79m缓慢扩宽，旧训练仅15–25m。
导航straight对照另存 outputs/custom_split_audit/20261103/custom_input_split_straight.json，原输入未改。


## 2026-10-30：联合模型 best 评估与分流过渡段诊断

用户已运行 `outputs/training_road_mixed_v1/eval_best_val`。总ADE0.360673m、EmptyBEV0.650670m。
路口正常/EmptyBEV ADE=0.593469/1.13216m；分流=0.107372/0.12677m。
路口entry ADE0.99896m、approach10 0.87494m；分流before10/during/after全程ADE=0.11018/0.13995/0.07199m。
新增 transition_region_summary.json 和 transition_region_per_sample.csv：按GT世界x落在[0,length_m]筛选过渡段，排除起点，
对有可评价点的样本等权。before10过渡段正常/EmptyBEV/Guide ADE=0.204670/0.262075/0.478282m；
during_mid=0.130967/0.122518/0.521814m，每组579样本。after5未来不含过渡段。
分流Guide自身可支持大部分修正，清空BEV仍有较高精度；不能将0.107m解释为已充分学习车道几何。
各场景最差样本另存 diagnostic_worst_intersection.png、diagnostic_worst_lane_transition.png。

## 2026-10-30：联合训练首轮已完成

已核实 `outputs/training_road_mixed_v1/history.jsonl`：epoch69早停，best为epoch44。
best混合验证3627样本（路口1890、分流1737），ADE/FDE=0.360673/0.511606m。
路口ADE/FDE=0.593469/0.889683m，Guide ADE=1.460471m；
分流ADE/FDE=0.107372/0.100228m，Guide ADE=0.496791m。
路口straight/left/right ADE=0.462179/0.702033/0.616197m，41–80m ADE=0.846790m。
最后epoch69路口ADE0.613956m、分流0.100197m，整体未刷新best，不建议直接延长。
本次尚未运行 evaluate.py，目录下未发现评估summary。下一步应评估best，并检查分流before/during/after及过渡段局部误差。
混合平均不能代表路口也达到0.36m；不能与旧输入上的旧成绩直接比较混合收益。

## 2026-10-30：准备批量分流 + 路口联合训练

新增 `Scenario.prepare_mixed_dataset`（重复 --source name=目录），保留来源 train/val/test World 归属，
拒绝重复 ID / World 冲突；可用 --refresh-perception intersection 从原物理 World 重导出路口感知，
统一 VCS 可见段内断裂，但保留 GT/Guide/pose/划分。旧支持诊断会清空。
新增 `configs/train_road_mixed_v1.json`，准备600分流 World（预计17280样本）+300已有路口（18900样本），
新数据输出 `outputs/road_mixed_v1`，新训练 `outputs/training_road_mixed_v1`。
架构与 Loss 沿用 V1，标准 FOV，无额外 FOV augmentation，从头训练；最多150轮、25轮无best早停、TensorBoard。
train.py 新增可选 validation_groups，每轮单独记录两类验证 ADE；选择仍按混合 val ADE。
evaluate.py 新增 scene_type 分组。生成器新增 --preview-worlds（默认5），避免批量画图。
完整执行命令见 `README_MIXED_ROADS.md`。完整600 World生成和正式训练尚未启动，由用户自行运行。
小样本流水线已实际跑通：3个分流World+6条真实路口样本，96条均可编码；37项相关测试通过。
批量生成允许全部漏检的空BEV样本，避免低质量场景在大批量时因非空断言中断。

## 2026-10-30：路沿+边线、可见段内感知断裂

用户澄清最外侧同时有路沿与车道线。生成器保留两侧 curb，并在各自内侧横向 0.25m
增加白色实线 lane_marking，过渡区仍无内部车道分隔线。
共享 `Scenario.build_dataset_v4.observe` 现先按 VCS 裁剪连续段，再在段内按原 break_rate
采样缺口；标记 `perception_convention=vcs_visible_segment_breaks_v2`。这也影响后续路口数据生成，
旧数据未重写。新批次 `outputs/lane_transition_preview_v4_visible_breaks`：5 World、162 样本，
GT/Guide 与 v3 逐条一致，感知 RNG 调用变化，不作为同噪声严格配对。
相关回归测试 35 项通过（含可见段断裂、VCS 旋转、固定 Guide、扩展感知及多进程训练流程）。

## 2026-10-30：分流 Guide 改为分段一次

用户要求降低分流 Guide 的平滑程度。`build_lane_transition.py` 新增
`--guide-shape piecewise_linear|smooth`，默认分段一次：入口/过渡直连/出口三段、四个世界锚点。
保留偏移和过渡位置扰动，在世界坐标固定后截取，GT/道路/感知不随该模式改变。
新预览 `outputs/lane_transition_preview_v3_linear`，5 World、162 样本；旧 v2 保留。
生成时显式读取 `perception_v4.example.json`。尚未训练。

## 2026-10-29 新增：增道/减道预览数据

用户确认场景为沿行驶方向从 x 条车道变成 x±n 条，而不限定为两条道路分叉。
新增 `Scenario/build_lane_transition.py`，说明 `README_LANE_TRANSITION.md`。
用户指出初版形态不符，已修订并生成 `outputs/lane_transition_preview_v2`：5 World、162 样本。
车道数前后均 1–4，变化 1–2；过渡长度 15–25m；过渡区内部无标线，只有道路边界。
每条连接 3 pose（before_10、during_mid、after_5）× clean/medium/hard，复用 V4 感知参数与函数。
格式兼容现有模型编码，固定 World Guide，nav=straight；World 隔离 3/1/1，未启动训练。
预览在 `previews/`，3×3 行质量列 pose，使用路口相同坐标与真值/感知配色，额外画 FOV。
当前为单侧增减道、平滑连接、GT 仅验证道路包络；
不代表已完成所有车道连接规则。先与用户确认形态，再扩充和制定路口+增减道混合训练。
`tests/test_lane_transition.py` 21 项测试通过，检查所有合法数量组合、无标线区域、输入编码、采样、固定 Guide 和确定性。
旧 `outputs/lane_transition_preview_v1` 保留，但不作为当前方案使用。

更新：2026-10-25。这是研究状态记录；文件内路径均相对项目根目录，换电脑后以实际文件为准。

## 当前目标和下一步

任务：离散 BEV polylines + 粗 Guide + Navigation，预测未来 80 m、81 个点的 GT path。
研究目标：探索能否把全程验证 ADE 降至 0.05 m。全程指标不取消，观测支持分区和同世界路段误差只作辅助诊断。

目前正在研究 Oracle（目标车道明确、相关物理几何无感知噪声）条件下的精度。完整训练已结束：训练 ADE 0.10514 m、验证 ADE 0.11923 m。下一步尚未执行：

1. 检查 32 点编码后的几何损失和高偏差样本。
2. 分解纵向/横向误差，以及按站点误差与到 GT 折线距离的差异。
3. 按 pose、转向、转弯段/出口段分组，定位误差集中在哪里。
4. 根据证据一次只改一个因素；暂未决定修改模型、输出参数化或 Loss。

不要将当前结果解释为模型精度的理论下限，也不要把 Oracle 与真实感知的差异全部归因于某一个因素。

## 沟通与运行约定

- 用中文，解释方法与证据，区分观察、关联和因果结论。
- 最近用户选择自己执行完整数据生成和训练；给出确实存在的脚本/配置和可执行命令。
- 不因交接而自动启动长训练或重建大数据；用户明确要求的评估可以执行。
- 已有数据、checkpoint、评估结果保留；新实验用新目录。
- 仓库若存在 .codegraph/，按照用户要求先用 CodeGraph 定位代码；此前检查没有该目录。

## 环境

原机器项目：/home/ygy/PathGenModel。
原 Python：/home/ygy/miniconda3/envs/toybev/bin/python；另一台机器需换为其 Python/conda 环境。
主要依赖：Python 3.10+、numpy、torch、shapely、matplotlib、pytest、tensorboard。
配置里的数据路径通常相对项目根目录，命令应在根目录运行。
一些旧 summary.json 记录原机器绝对路径；若脚本依据它们定位文件，换电脑后要核实路径，不能盲目使用。

## 数据及划分

当前常规数据：outputs/v4_world300_inside13。
300 World，18900 样本；7 pose × 3 转向 × 3 感知质量 = 每 World 63 条。
pose：approach_30、approach_20、approach_10、entry、inside_8、inside_13、exit_5。
inside_13 为从 GT entry 沿路线前进 13 m；若已经到达/越过 exit，则跳过。当前 300 World 均生成了该位置。
生成器：Scenario/build_dataset_v4.py；共享 pose/稳定 RNG ID：Scenario/build_dataset.py。
新增 pose RNG ID 为 6，旧 pose 编号没有改变；补充感知脚本也支持它。

划分位于 audit/train.jsonl、val.jsonl、test.jsonl，以 World 为单位，240/30/30 World；不要按行重新拆分。
GT 与 Guide 在同一 route 的不同感知质量下保持一致。

Guide 修过一个重要 bug：原来每个 pose 做 ref -= ref[0] 会抹掉出口常量偏差。
现为同一 guide_world 只截取并刚体变换；Guide 起点允许非零，GT 和预测起点仍固定为零。
约定名 world_fixed_ego_transform_v1。相关脚本 Scenario/reexport_guides.py。
旧数据 outputs/v4_world300_fixedguide 为六 pose 版本，不要与 inside13 版本混淆。

## 当前模型与 Loss

Model/pathgen_model_v1.py：point feature 6 维，polyline attr 10 维，最多 32 条线，每条 32 点。
D=128，4 heads，spatial encoder 2 层，decoder 2 层，dropout=0.1。
Polyline 点编码/池化成 token，Guide 点编码为 query，融合 Navigation，输出 Guide + XY residual。
已有 query—polyline 最近线段相对几何，进入 per-head attention additive bias；不是局部几何 value 的直接聚合。
没有做过坐标归一化改造、Frenet 输出改造或 Residual Gate。
GT/预测共 81 点，间隔 1 m，起点强制为零。
Model/losses.py：位置 SmoothL1 + 0.1 × 二阶差分对 GT 的匹配；位置项排除起点。
停止线/斑马线也由 polyline encoder 处理，类型由 attributes 区分。

## Oracle 实验：现在最重要

说明：README_ORACLE.md。
导出脚本：Scenario/export_oracle_lanes.py。
数据：outputs/oracle_lane_v1。

Oracle 用真实入口/目标出口 lane ID 筛选物理边界，只保留两侧相关边界，隐式明确目标身份。
坐标来自独立 World physical 元素；不把 GT path、GT 中心线或 GT residual 做输入。
保留原粗 Guide、GT、Navigation；无感知噪声、断裂或漏检，ego 四向范围 100 m。
仍保持原模型的 32-token / 32-point 表达。无其他车道候选、停止线或斑马线。
原三种感知质量变成相同输入，因此只保留原 clean 一份并标为 oracle，避免重复三份。
World 划分继承原数据：train 5040、val 630、test 630。

小拟合实验：
- 配置 configs/train_oracle_fit.json。
- 4 个训练 World（84 样本），2 个独立验证 World。
- 1000 epochs，无 early stop，LR 1e-4 保持不变。
- outputs/training_oracle_v1/fit/last.pt 在训练集 eval 模式 ADE 0.0579976 m，FDE 0.0609065 m。
- 训练时 dropout 开启，日志 train ADE 不能直接当作拟合极限；拟合诊断要评估 last.pt 的训练集。

完整 Oracle 实验：
- configs/train_oracle_full.json；最多 300 epochs，50 epochs 无 best 早停。
- 第 138 个 epoch 早停，best.pt 为 epoch 88。
- outputs/training_oracle_v1/full/best.pt。
- eval_train/summary.json：ADE 0.1051433 m，FDE 0.1220106 m。
- eval_val/summary.json：ADE 0.1192326 m，FDE 0.1415428 m，ADE P90 0.2257340 m。
- val 分段 ADE：1–20 m 0.0816188，21–40 m 0.1179912，41–80 m 0.1386602。
- 训练/验证 ADE 差约 1.4 cm，尚不能把剩余问题主要归因于明显过拟合。

已读取 outputs/oracle_lane_v1/export_report.json：32 点编码边界相对原物理折线的最大偏差 0.080504 m，P90 0.035158 m，134 条样本最大偏差超过 5 cm。
这尚未做深入分组，且边界偏差不是 path ADE 的严格下限！这是下一步分析的重要线索。

评估命令示例（输出已存在时需换目录）：

```bash
python evaluate.py --checkpoint outputs/training_oracle_v1/full/best.pt --no-empty-bev --output-dir outputs/training_oracle_v1/full/eval_val_new
python evaluate.py --checkpoint outputs/training_oracle_v1/full/best.pt --split train --no-empty-bev --output-dir outputs/training_oracle_v1/full/eval_train_new
```

## 已做研究与证据

### Mixed 与扩展感知

configs/train_context_control_inside13.json、train_context_mixed_inside13.json。
输出 outputs/context_comparison_inside13/control 和 mixed。
Mixed 按 epoch/sample 确定性抽样：50% 原输入，其他各 12.5% 补至多 4/8/12/全部 retained tokens。
训练 cache：outputs/context_train300_inside13，seeds 1031/1032。
扩展感知不重新生成旧观测，只在原 FOV 外补充，保持质量参数，最多填充 32-token 空位。
验证 cache：outputs/context_comparison_inside13/fixed_val_supplements，seeds 724/725/726。
不得按 GT 选择正式训练/推理的补充 token；Oracle 是明确标记的特权诊断例外。

Inside13 Mixed best 为 epoch 38，第 58 轮早停。全量 val 1890 样本：
- 原始 ADE/FDE：0.597270 / 0.905977 m。
- 三扩展种子均值：0.585358 / 0.885791 m。
- pose_context_evaluation/by_pose.csv 含完整按 pose 原始/扩展结果。
- entry 最难，扩展 ADE 0.9863；inside_8 0.4737；inside_13 0.2954；exit_5 0.1065。
- 单看原六 pose，新 Mixed 原始 ADE 0.6449（上一版 0.6552），扩展 0.6337（上一版 0.6236）；新增容易 pose 会改变整体平均，不能直接归因比较。
- 当时用户说本轮 Control 仍在训练；其最新状态需在新会话重新检查。

上一轮六 pose 公平四条件结果：outputs/context_comparison/evaluation/summary.json。
Control 原始/扩展 ADE 0.6437/0.7591；Mixed 原始/扩展 0.6552/0.6236（三扩展种子均值）。
排除 Guide lane 歧义的 738 条子集：control 0.6331/0.7115；mixed 0.6136/0.5928。
相关目录 outputs/context_comparison/clear_guide_evaluation；没有证明扩展信息对每条样本有益。

### Guide 与车道身份歧义

audit_guide_lane_ambiguity.py，报告 outputs/context_comparison/guide_lane_audit/findings.md。
旧 90 条路线中，61 多车道：44 target_clear，6 near-tie/mixed，11 更靠近其他车道。
在目标明确组仍存在选错车道问题；不能全部归罪于 Guide。
重要案例 World 32019345 left entry：Guide 几乎在两车道分隔线上，GT 与预测各在一个车道。
生成器将入口/出口车道组合打乱，选可行组合，没有固定“左转一定去某出口 lane”的规则。
补入理想出口边界（不显式标注目标身份）只小幅改善稳定反向案例，报告 outputs/context_comparison/exit_boundary_oracle。

### 同一世界出口路段跨 Pose 对照

diagnose_pose_progression.py，outputs/context_comparison_inside13/mixed/pose_progression/findings.md。
270 组相同 World/nav/quality，entry/inside8/inside13 均评价出口后 3–30 m 的相同 28 个世界站点。
原始同段误差 1.261 → 0.559 → 0.327 m；扩展 1.230 → 0.541 → 0.312 m。
不是仅因困难转弯退出移动 80 m 窗口造成的下降，但前瞻距离也缩短了。
左转扩展两侧边界支持 44.8% → 36.2% → 48.0%，误差 1.584 → 0.743 → 0.438 m。
32 条左转三元组在共同窗口内都没有任一侧目标边界支持，仍平均从 1.785 降至 0.503 m。
因此不能把进入路口后的改善只归因于新增出口边界数量；姿态/位置、候选 lane 关联、Guide 截取等共同变化。
支持阈值为 1.25m，基于实际 retained / 32点编码几何与诊断 source IDs，不等于严格信息可观测性。

## 迁移文件

最省事是复制整个项目（含 outputs），再在新机器重建 Python 环境。
若只继续当前 Oracle 研究，至少携带：
- 所有代码、configs、tests、README 文档及本文件。
- outputs/oracle_lane_v1（数据/World/routes/划分/export_report）。
- outputs/training_oracle_v1（checkpoint/config/history/评估结果）。
- 若要继续真实感知对照：outputs/v4_world300_inside13、context_train300_inside13、context_comparison_inside13。
- 若要复核旧歧义结论：outputs/context_comparison 及其引用的数据/预测；可按报告路径补充迁移。

代码和 Markdown 可用 Git 管理；大数据、checkpoint 和未跟踪的 outputs 需确认实际同步，不能仅复制代码。
新会话启动后先核实上述文件是否存在，不应根据本记录假定所有文件已经迁移。

## 2026-10-26：自定义冲突测试后的数据覆盖审计

新报告：outputs/guide_boundary_audit/20261026_132823/findings.md。
脚本：scripts/audit_guide_boundary_conflicts.py。仅分析现有train/val与保存的Mixed预测，没有启动训练。
训练去quality重复后的转弯窗口：右转374/1440个pose输入Guide明显出物理道路（25.97%，来自63条独立路线）；不是完全缺乏冲突。
验证全80m三quality：Guide出道路255条，预测仍出道路19条；右转150条中5条仍出道路。clean右转转弯窗口47条冲突，预测明显出道路0条。
因此原分布上多数能修正；自定义失败不能解释成模型普遍不会规避。冲突/非冲突验证ADE约0.598/0.597m，尚不能把整体误差主要归因于冲突。
每World/nav仅一条Guide，Mixed只补tokens，没有同道路多Guide或后方元素干预的一致性训练。
同世界出口后3–30m跨entry/inside8/inside13：entry冲突39组三元组（13 Worlds）平均跨pose横向max-min为0.815m；信息/姿态/噪声同时变化，不能当作单因素因果结果。原Oracle精度诊断目标仍未完成。

## 2026-10-26：已执行固定道路、多Guide专项评测（第一步）

说明：README_GUIDE_CONFLICT_BENCHMARK.md；脚本scripts/evaluate_guide_conflicts.py；完整报告outputs/guide_conflict_benchmark/val30_mixed_v1_20261026/findings.md。
30验证World/60左右转路线，3pose×3quality=540基础输入，每个13种Guide；排除54条自交Guide案例后有效6966条。道路/GT/感知输入固定，世界Guide变形后跨pose截取；未训练、未重建World，test未用。
Mixed专项基准原始Guide ADE0.891m、明显出道路3.33%；Guide转弯附近提前/延后3m后ADE1.766/1.975m、出道路20.74%/21.48%。这是偏困难子集与合成干预，不能替换全验证0.597m。
可见编码路沿冲突子集：提前3m为41/141仍穿越，延后3m为19/62；clean输入也显著退化。支持Guide误差泛化/几何约束不稳，不能单独认定网络或Loss原因。
仅原始与±3m三Guide，出口平均有符号横向误差跨Guide范围均值1.668m。更极端干预含车道身份歧义，须结合报告解释。
6项测试通过；原始预测复现最大坐标差7.63e-6m；GT无出道路、非Guide哈希一致、train/val World无交集。配方、预测、逐例指标与可infer重放示例均已保存。尚未执行第二步训练。

## 2026-10-26：下一轮训练入口已准备，尚未启动正式训练

README_GUIDE_TRAINING.md给出完整命令。用户自行运行：先python scripts/build_training_guide_cache.py，再python train.py --config configs/train_guide_aug_v1.json --no-tensorboard。
新模块Dataset/guide_augmentation.py：左右转50%原Guide、50%从每World/nav八条连续变体采样，直行不变；每epoch仍15120条，不增训练步数。只替换Guide张量，BEV/GT/nav不变；跨pose共享世界Guide。缓存仅使用train240Worlds，输出outputs/guide_train_cache_v1。
新增TrainConfig.guide_augmentation，缓存指纹进入checkpoint splits，验证不增强；与FOV增强互斥，保持单因素。既有训练配置默认行为保持不变。
configs/train_guide_aug_v1.json从随机seed42开始，模型/Loss/优化/最大100epoch与原Control一致，输出outputs/guide_training_v1/aug；严格重跑Control配置train_guide_control_v1.json输出独立control目录。旧Mixed不作为无缝续训起点；主要单因素对照是Control，旧Mixed只作参考。
训练后先evaluate.py导出新模型原验证predictions.npz，再向scripts/evaluate_guide_conflicts.py传新checkpoint和对应--baseline-predictions，复用固定压力测试条件。不能用旧Mixed的baseline档案验证新权重。
相关测试15通过、1项因缺TensorBoard跳过；双worker与精确续训测试在允许本地IPC的执行环境通过。没有生成正式完整缓存或启动完整训练。

## 2026-10-26：Guide增强第一轮训练已由用户完成并复核

增强outputs/guide_training_v1/aug/best.pt为epoch78，第98轮早停。原验证ADE0.624594m/FDE0.946486m。新增同条件Control评估outputs/guide_training_v1/control_reference_eval_val_20261026（ADE0.607839m），及Control压力评测outputs/guide_conflict_benchmark/control_reference_v1。
比较报告outputs/guide_training_v1/comparison_20261026/findings.md；两组初始模型SHA相同，train/val指纹及主要超参数相同，FOV增强都为null，仅Guide增强、日志开关和输出目录不同。
固定6966压力案例：Control/增强 ADE3.0170/1.3283m，明显出道路32.29%/13.71%，可见路沿穿越15.50%/1.92%，可见冲突修复57.85%/96.66%。常规ADE退化约1.68cm（2.76%）；exit_5由0.0838到0.1724m，是主要净退化贡献，inside8/13亦退化。右转整体亦退化，不能仅引用整体approach10改善。
自定义81.85m Guide、固定停止线/斑马线、同时延长右侧车道线/路沿：路沿终点7/10/13/16/19m，Control后三次穿越，增强五次无穿越。结果在outputs/guide_training_v1/custom_curb_followup_20261026；增强有先左偏再右转绕行，无GT不能认定正确路径。
待验证下一轮：仅在approach30/20/10及entry启用Guide增强，inside8/13/exit5保留原Guide，其余条件不改，以检验出口精度恢复。该方案尚未实现/训练。Oracle精度诊断仍待执行。

## 2026-10-27：Δs/Δl与纯横向输出诊断完成，未改模型

报告outputs/lateral_output_diagnostic/20261027_v1/findings.md；脚本scripts/diagnose_lateral_output.py。原训练5040/验证630唯一几何pose，核验真实感知全部15120/1890条GT/Guide相同。
保持现有同序号输出p_i=Guide_i+Δl_i*n_i时，GT理想Δl的全80m ADE下限=train0.6761m、val0.5889m；val左/右/直行0.8647/0.7626/0.1394m。故不能直接删切向自由度达到5cm。原Oracle0.11923m，删去其预测切向分量后ADE0.61269m。
val mean abs GT所需Δs/Δl=0.5889/1.1945m；纵向占残差平方和31.33%。增强训练池纯横向下限2.0735m，当前训练混合期望1.0581m。
重要区别：GT理想横向修正曲线与GT的双向最近折线平均距离仅0.02525m，沿程对应是重要因素。投影曲线长度足80m的337/630子集上，按弧长重采样ADE从0.7207降到约0.05756m；另293条长度不足，不能当全验证性能或理论最优。
建议当前逐点任务保留Δs；l(s)+重采样是值得单独研究的不同方案，需先解决参考曲线覆盖与起点等。5项诊断测试通过，未训练/未改变任何模型、Loss或策略。
