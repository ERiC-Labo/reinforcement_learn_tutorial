import gym
import robo_gym

robot_address = '127.0.0.1'

# initialize environment
# env = gym.make('NoObstacleNavigationMir100Rob-v0', rs_address=robot_address)
env = gym.make('EndEffectorPositioningURSim-v0', ur_model='ur3', ip=robot_address)

num_episodes = 1

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        state, reward, done, info = env.step(env.action_space.sample())