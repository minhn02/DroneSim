from arty_wrapper import ArtyAgent

arty = ArtyAgent()

obs = []

for i in range(12):
    obs.append(i)

inputs = arty.act(obs)

print(inputs)