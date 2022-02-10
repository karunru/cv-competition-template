#!/usr/bin/env bash
mkdir "log"
touch log/.gitkeep
mkdir "weights"
touch weights/.gitkeep
mkdir "preds"
touch preds/.gitkeep
mkdir "dataset"
touch dataset/.gitkeep
mkdir "notebook"
touch notebook/.gitkeep
mkdir "config"
touch config/.gitkeep

# template
cp -r ../cv-competition-template/augmentations ./
cp -r ../cv-competition-template/losses ./
cp -r ../cv-competition-template/models ./
cp -r ../cv-competition-template/optimizer ./
cp -r ../cv-competition-template/sampling ./
cp -r ../cv-competition-template/utils ./
cp -r ../cv-competition-template/validation ./
cp -r ../cv-competition-template/table ./
cp -r ../cv-competition-template/config ./
cp -r ../cv-competition-template/*.py ./
cp -r ../cv-competition-template/*.sh ./
cp ../cv-competition-template/pyproject.toml ./
cp ../cv-competition-template/.gitignore ./

# idea settings
project_name=$(basename `pwd`)
cp -r ../cv-competition-template/.idea ./
mv .idea/cv-competition-template.iml .idea/"${project_name}.iml"
sed -i -e "s/cv\-competition\-template/${project_name}/g" .idea/modules.xml
sed -i -e "s/cv\-competition\-template/${project_name}/g" .idea/deployment.xml
sed -i -e "s/cv\-competition\-template/${project_name}/g" .idea/remote-mappings.xml
