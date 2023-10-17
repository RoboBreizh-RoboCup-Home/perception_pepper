from setuptools import find_packages, setup
import os

package_name = 'perception_pepper'

setup(
    name=package_name,
    version='0.0.0',
    packages=['PoseDetection', 'ObjectDetection', 'FaceDetection', 'AgeGenderDetection', 'ColourDetection', 'GlassesDetection', 'example_and_tests', 'ros_node'],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maelic',
    maintainer_email='teoneau@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_demo = scripts.example_and_tests.pepper_inference_opencv:main',
            'pose_demo = scripts.example_and_tests.demo_pose:main',
        ],
    },
)