import json, datetime

with open("live_moderation_log.json", "r") as f:
    logs = json.load(f)

print("\n🧠 Debate Moderation Summary\n" + "-" * 40)
print(f"Total entries: {len(logs)}\n")

actions = {"warn": 0, "mute": 0, "remove": 0, "no_action": 0}
for log in logs:
    actions[log.get("action", "no_action")] = actions.get(log.get("action", "no_action"), 0) + 1

print("Action Counts:")
for k, v in actions.items():
    print(f"  {k:10}: {v}")

print("\nRecent Logs:")
for log in logs[-5:]:
    print(f"[{log['timestamp']}] {log['user']} → {log['text']} ({log['action']})")

print("\n✅ Report generated:", datetime.datetime.now().isoformat())
