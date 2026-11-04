# 达芬奇器械与脂肪外观

本次仅替换器械几何/外观和组织材质，并验证替换后的夹取、拉伸与单刀片流程。

- 夹钳：使用 MIT 许可 dvrk_description 的 Classic 8 mm Cadiere，固定提交 2a29763608fd619082dec112f07014256cc62c22；保留镂空、齿形和腕部零件。原始网格以米计，适配到本项目厘米坐标。默认钳口长度19.9 mm、宽5.2 mm，轴杆外径8.32 mm（源模型测量）。
- 单刀片：参考 Snap-fit Scalpel，使用公开腕部模型和本地简化刀片/装夹件。没有找到该刀片的公开原厂CAD，因此明确称外形适配；单刀片接触仍为有限刃口加刀背/轴杆代理。
- 接触：新钳口与接触包围盒从同一组变换后的顶点取得；镂空与细齿仍按接触包络近似。显示腕部处于保守轴杆接触包络内。
- 脂肪：ImageGen生成纹理，仅应用于可操作块与新切面；背景保留肝组织色调。纹理绑定材料坐标，拉伸时跟随组织。保留原材质资产；脂肪外观不代表材料参数已按脂肪标定。
- 验证：单位/钳口配准检查，初始及夹持/切割近景渲染，完整单刀片流程与退刀重叠阈值检查。录制原始画面，不加侧栏。

来源：https://github.com/zixingjiang/dvrk_description
参考目录：https://pdf.medicalexpo.com/pdf/intuitive-surgical/instrument-accessory-catalog/75060-101907.html

## 纹理上传与几何检查

生成图为1254像素宽。当前OpenGL上传器使用默认4字节行对齐，RGB法线纹理的每行字节数不满足该对齐，导致首帧在glTexImage2D处访问异常。原型将这张法线纹理显式编码为RGBA后完成首帧渲染，保留原始贴图尺寸和像素，不修改Genesis核心。

13项CPU检查通过，新增检查覆盖Cadiere默认尺寸、腕部在夹钳胶囊/刀杆盒中的包络，以及0°/45°安装姿态。

## 轨迹适配

8 mm腕部加入完整保守包络后，原0°深切轨迹在进刀末端出现0.270 mm刀背/刀杆重叠，超过0.2 mm验收阈值，该次录像标记失败并保留在demo-final。默认演示切入目标从原生y=-0.15 cm改为y=+0.15 cm，为较粗腕部留出空间；接触阈值不变，重新运行完整流程。45°、目标y=+0.15 cm的独立单次检查通过，记录在yaw45-check，不能据此声称原0°深切或多次切割全部通过。

后续独立复核发现demo-v2虽满足接触与夹持阈值，但在incise、withdraw、hold阶段各存在1条非流形边，因此完整验收仍失败，追加acceptance-review.json纠正原报告的范围。run.py现已把每个阶段的边界、非流形、重复面、退化面和绕序检查纳入成功判定；默认刀向改为45°，重新验证与录制。

## 最终验证（2026-10-27）

- `default-verified`：新默认45°流程565步通过，20个夹持点持续保持，中位提拉2.761 mm，82条断边与270个切面三角形；刀背/刀杆最大重叠0.0053 mm，退刀新增断边0。8个阶段的退化面、开放边界、非流形边、重复面均为0，绕序一致。
- `demo-yaw45`：45°单次切割录像567步通过，71条断边（其中57条属于材料连接），218个切面三角形，提拉3.909 mm，夹持误差最大0.096 mm，刀背/刀杆重叠0，退刀新增断边0。保存状态的10个检查点均通过独立拓扑审计。
- 这两次运行使用同一DLL，散列为`2632a926b385d9c9a81d5f7d7ebb75fe0501917a9cf83d0c19d3a29c62af961b`；控制时序不同，原生GPU求解也并非确定性复现，不要求断边数完全相同。录像的峰值反力为27.53 N，默认流程为2.34 N，尚未做组织/器械力学标定；本轮通过代表当前几何与流程验收，不代表力学真实性已验证。
- 最终原视频：`outputs/davinci-fat/demo-yaw45/endoscopic_demo_h264.mp4`，1280×960，15 fps，283帧，18.867秒，无侧栏讲解。ffmpeg全片解码与OpenCV逐帧读取通过；抽查夹持、进刀、退刀画面。
- 当前切口仍能看到物理网格造成的折面。纹理和Cadiere细节已接入；没有据此声称任意方向或重复切割已稳定。0°失败样本保留。

复现录像：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.stress_cutting --yaw 45 --cycles 1 --video --render-every 2 --output endoscopic_tissue\outputs\davinci-fat\new-demo
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.audit_repeated endoscopic_tissue\outputs\davinci-fat\new-demo
```

`new-demo`必须为尚不存在的目录。
