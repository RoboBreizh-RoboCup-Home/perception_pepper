from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['perception_utils', 'ros_node', 'task_node','ObjectDetection', 'PoseDetection', 'ColorsDetection', 'AgeDetection', 'Camera'],
    package_dir={'':'scripts',
                 'perception_utils': 'scripts/perception_utils',
                 'ros_node': 'scripts/ros_node',
                 'task_node': 'scripts/task_node',
                 'Camera': 'scripts/Camera',
                 'ObjectDetection': 'scripts/models/ObjectDetection',
                 'PoseDetection': 'scripts/models/PoseDetection',
                 'ColorsDetection': 'scripts/models/ColorsDetection',
                 'AgeDetection': 'scripts/models/AgeDetection'}
)
setup(**d)