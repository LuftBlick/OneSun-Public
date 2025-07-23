# OneSun
A physics-constrained, self-calibrating, data-driven system
with a common solar spectrum for homogeneous trace gas retrieval networks


## Getting started
first the usual maybe...
python -m venv .venv 
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=.venv


You can create your dataset by running the l_data_loader.ipynb notebook. 
This notebook will download PGN L0 files for a given instrument in a given time range, and create i pickle file with the dataset.

### Getting the data
The l0 files for creating the dataset can be downloaded from Google Drive:
https://drive.google.com/drive/folders/104TFksmW6e2tXw4pf1vChOqkcyZgrTsE


## Docker
You can run the OneSun system in a Docker container. The Dockerfile is located in the root directory of the repository.
To build the Docker image, run the following command in the root directory of the repository:

```bash
docker compose up
```