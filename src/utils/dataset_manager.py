import os
import json
import torch
from pathlib import Path
import hashlib
from datetime import datetime

class DatasetManager:
    """
    Manages centralized dataset storage and retrieval
    """
    def __init__(self, base_dir=None):
        # Use environment variable if provided, else default
        env_base_dir = os.environ.get("ONESUN_DATASET_DIR")
        if base_dir is None:
            base_dir = env_base_dir if env_base_dir else "data/processed_datasets"
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[2]
        candidate = Path(base_dir)
        if not candidate.is_absolute():
            self.base_dir = (project_root / candidate).resolve()
        else:
            self.base_dir = candidate
        print(f"DEBUG: DatasetManager base_dir: {self.base_dir}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "dataset_index.json"
        self._load_index()

    
    def _load_index(self):
        """Load dataset index or create if not exists"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "datasets": {},
                "last_updated": datetime.now().isoformat()
            }
            self._save_index()
    
    def _save_index(self):
        """Save dataset index"""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=4)
    
    def save_datasets(self, train_dataset, val_dataset, config, friendly_name=None, metadata=None):
        """
        Save datasets centrally, overwriting any existing datasets with the same name
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary used to create the datasets
            friendly_name: Optional human-readable name for the dataset
            metadata: Optional additional metadata
            
        Returns:
            Dataset ID (friendly name) for the saved datasets
        """
        # If no friendly name is provided, generate one
        if friendly_name is None:
            ds_name = Path(config.get("dataset_name", "unknown")).stem
            train_start = config.get("train_date_range", ["", ""])[0].replace("-", "")
            train_end = config.get("train_date_range", ["", ""])[1].replace("-", "")
            friendly_name = f"{ds_name}_train{train_start}-{train_end}"
        
        # Check if we're overwriting an existing dataset
        if friendly_name in self.index["datasets"]:
            print(f"Overwriting existing dataset: {friendly_name}")

        # Create/overwrite dataset directory
        dataset_dir = self.base_dir / friendly_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save datasets
        if train_dataset is not None:
            torch.save(train_dataset, dataset_dir / "train_dataset.pt")
        
        if val_dataset is not None:
            torch.save(val_dataset, dataset_dir / "val_dataset.pt")
        
        # Save metadata
        dataset_metadata = {
            "config": {
                "dataset_name": config.get("dataset_name", ""),
                "train_date_range": config.get("train_date_range", ["", ""]),
                "val_date_range": config.get("val_date_range", ["", ""]),
                "source_file": Path(config.get("pickle_path", "")).name,
            },
            "created_at": datetime.now().isoformat(),
            "user_metadata": metadata or {}
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(dataset_metadata, f, indent=4)
        
        # Update index
        self.index["datasets"][friendly_name] = {
            "path": str(dataset_dir),
            "created_at": dataset_metadata["created_at"],
            "config": dataset_metadata["config"]
        }
        self._save_index()
        
        return friendly_name
    
    def load_dataset(self, dataset_id=None, config=None, dataset_type="val"):
        """
        Load dataset by ID (friendly name) or try to find by config
        
        Args:
            dataset_id: Optional dataset ID (friendly name)
            config: Optional config to use for generating friendly name
            dataset_type: 'train' or 'val'
            
        Returns:
            The loaded dataset or None if not found
        """
          
        # If no ID provided but config is, generate a name to try
        if dataset_id is None and config is not None:
            ds_name = Path(config.get("dataset_name", "unknown")).stem
            train_start = config.get("train_date_range", ["", ""])[0].replace("-", "")
            train_end = config.get("train_date_range", ["", ""])[1].replace("-", "")
            dataset_id = f"{ds_name}_train{train_start}-{train_end}"
        
        if dataset_id not in self.index["datasets"]:
            print(f"Dataset ID '{dataset_id}' not found in index.")
            print("Available datasets:",len(self.index["datasets"]))
            for ds_id in self.index["datasets"]:
                print(f" - {ds_id}")
            return None
        
        #dataset_path = Path(self.index["datasets"][dataset_id]["path"]) / f"{dataset_type}_dataset.pt"
        dataset_folder = self.base_dir / dataset_id
        dataset_path   = dataset_folder / f"{dataset_type}_dataset.pt"
        
        print(f"DEBUG: Loading '{dataset_type}' split for dataset_id='{dataset_id}' from path:")
        print(f"       {dataset_path.resolve()}")

        if not dataset_path.exists():
            return None
        
        try:
            return torch.load(dataset_path)
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {str(e)}")
            return None
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.index["datasets"].items())
    
    def get_dataset_metadata(self, dataset_id):
        """Get metadata for a dataset"""
        if dataset_id not in self.index["datasets"]:
            return None
        
        metadata_path = Path(self.index["datasets"][dataset_id]["path"]) / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)