from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

requirements = ['numpy>=1', 'tensorflow>=1']

setup(
    name='baakbrothers',
    version='0.0.2',
    url='https://github.com/yukikitayama/baakbrothers',
    author='Yuki Kitayama',
    author_email='yk2797@columbia.edu',
    description='Reinforcement learning package',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requirements=requirements
)