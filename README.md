# reinforcement_learn_tutorial
まずopenai_gym_ros/training/turtlebot2_training/config/turtlebot2_wall_params.yamlを開いて以下を訂正
```
turtlebot2: #namespace
    task_and_robot_environment_name: 'MyTurtleBot2Wall-v0'
    ros_ws_abspath: "/home/ericlab/ros_package/saikou_ws"   ##自分がコンパイルしたパッケージのワークスペースのパス
    running_step: 0.04 # amount of time the control will be executed
    pos_step: 0.016     # increment in position for each command
    
    #qlearn parameters
    alpha: 0.1
    gamma: 0.7
    epsilon: 0.9
    epsilon_discount: 0.999
    nepisodes: 500
    nsteps: 10000
    running_step: 0.06 # Time for each step
```
依存パッケージをいくつかインストール
```
pip3 install gym
pip3 install gitpython
```
次に以下を実行
```
roslaunch turtlebot2_training start_training_wall.launch
```
