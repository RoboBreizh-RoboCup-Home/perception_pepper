from setuptools import find_packages, setup
import os

package_name = 'perception_pepper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['tests']),
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