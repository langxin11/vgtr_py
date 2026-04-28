import json
from pathlib import Path

DESCRIPTIONS = {
    "claw.json": "机械爪 - 具有抓取能力的变几何桁架结构。",
    "crawling-bot.json": "爬行机器人 - 通过步态协调实现地面移动。",
    "crawling-lobster.json": "龙虾机器人 - 仿生龙虾形态的爬行机构。",
    "dancing-bot.json": "跳舞机器人 - 能够执行复杂律动动作的桁架机器人。",
    "dog-standing.json": "站立小狗 - 四足站立姿态的基础桁架模型。",
    "example.json": "基础示例 - 展示 VGTR 数据格式的基本组成。",
    "GeoTrussRover.json": "变几何桁架漫游者 - 高性能移动探测平台。",
    "globe-fish.json": "河豚机器人 - 能够改变体积形态的仿生桁架。",
    "living-flower-I.json": "活体花卉 I - 模拟植物开合动作的变几何机构。",
    "living-flower-II.json": "活体花卉 II - 进阶版变几何植物模型。",
    "running-dog.json": "奔跑小狗 - 具有动态奔跑能力的四足桁架机器人。",
    "static-turtle.json": "静态海龟 - 稳定的多足支撑结构示例。",
    "transformer-rocket-I.json": "变形火箭 I - 能够实现形态折叠的桁架系统。",
    "transformer-rocket-II.json": "变形火箭 II - 进阶变形逻辑演示。",
    "walking-bot-with-gait.json": "步态行走机器人 - 包含预设步态脚本的移动平台。",
    "wandering-bot.json": "漫游机器人 - 随机移动测试平台。",
}

CONFIGS_DIR = Path("configs")


def main():
    for filename, desc in DESCRIPTIONS.items():
        file_path = CONFIGS_DIR / filename
        if not file_path.exists():
            print(f"Skipping {filename}, not found.")
            continue

        print(f"Annotating {filename}...")
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Insert description at the top (dict insertion order matters for JSON)
            new_data = {"description": desc}
            new_data.update(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
