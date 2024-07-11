import os

from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import ExecuteProcess

from launch_ros.actions import Node



def generate_launch_description():

    #Arguments
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Inicilize rviz or not'
    )

    realsense = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('realsense2_camera'),'launch','rs_launch.py'
                )]),
                launch_arguments = {
                   'align_depth.enable' : 'true',
                }.items()
    )

    realsense_params = LaunchDescription([
            ExecuteProcess(
                cmd=['ros2', 'param', 'load', '/camera/camera', 'src/yolo_pose/config/camera__camera.yaml'],
                output='screen'
            ),
        ])

    yolo = Node(package='yolo_pose', executable='inference',
                        output='screen')
    
    images = Node(package='yolo_pose', executable='draw_keypoints',
                    output='screen')
    
    distance = Node(package='yolo_pose', executable='distance',
                    output='screen')
    
    # Launch RViz2 with a configuration file
    rviz = Node(package='rviz2', executable='rviz2',
                arguments=['-d', os.path.join(get_package_share_directory('yolo_pose'), 'configs', 'view_bot.rviz')],
                condition=IfCondition(LaunchConfiguration("rviz"))
    )

    return LaunchDescription([
        rviz_arg,
        realsense,
        realsense_params,
        yolo,
        images,
        distance,
        rviz,
    ])