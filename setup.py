from setuptools import find_packages, setup

package_name = 'ros2_dinosaur'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Jonas Loos',
    maintainer_email='33965649+JonasLoos@users.noreply.github.com',
    description='object detection and position estimation using a camera feed and depth information',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dinosaur = ros2_dinosaur.dinosaur:main',
        ],
    },
)
