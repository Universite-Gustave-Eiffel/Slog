import setuptools
import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def parse_requirements_file(filename):
    # copied from networkX
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
    return requires

setuptools.setup(
    name="slog",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    extras_require = {"default": parse_requirements_file("requirements.txt")}
)

