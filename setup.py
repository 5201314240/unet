from setuptools import setup, find_packages  

setup(  
    name='unet',  # 项目名称  
    version='0.1.0',          # 项目版本  
    author='52013141240',       # 作者名称  
    author_email='2833980897@qq.com',  # 作者邮箱  
    description='A brief description of your project',  # 项目描述  
    long_description=open('README.md').read(),  # 项目详细描述，通常从 README 文件读取  
    long_description_content_type='text/markdown',  # 描述内容类型  
    url='https://github.com/52013141240/unet',  # 项目主页  
    packages=find_packages(),  # 自动查找项目中的包  
    classifiers=[  # 分类器，帮助用户找到你的项目  
        'Programming Language :: Python :: 3',  
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',  
    ],  
    python_requires='>=3.6',  # Python 版本要求  
    install_requires=[  # 项目依赖  
        'numpy',  # 示例依赖  
        'opencv-python',  # 示例依赖  
    ],  
)
