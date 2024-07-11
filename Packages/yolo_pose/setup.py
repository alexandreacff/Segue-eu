from setuptools import setup
import os 
from glob import glob

package_name = 'yolo_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.engine'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='athome-viper',
    maintainer_email='home.pmec@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference = yolo_pose.yolo_inference:main',
            'draw_keypoints = yolo_pose.yolo_image:main',
            'distance = yolo_pose.pose_distance:main',
            'distance_v2 = yolo_pose.pose_distance_v2:main',
            'distance_v3 = yolo_pose.pose_distance_v3:main',
            'track = yolo_pose.yolo_tracker:main',
        ],
    },
)
