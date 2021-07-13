from environments.isc_environments.SimpleISC import SimpleISC


if __name__=="__main__":
    env = SimpleISC(mode="DISCRETE")

    obs, reward, done, info = env.reset(), 0, False, None
    total_reward = 0
    step = 0

    while not done:

        print(f"Step {step} | Soc: {obs[4]}, speed: {obs[5]}, reward: {reward}")
        action = input("Action (0-2) ->")
        action = 1 if action == "" else int(action)
        print()
        step += 1

        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Obs: {obs}")
    print(f"\n\nTotal reward: {total_reward}, info: {info}")