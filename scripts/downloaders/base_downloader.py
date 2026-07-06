from pathlib import Path
from logger import logger

class BaseDownloader:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.downloaded = 0
        self.skipped = 0
        self.failed = 0

    def ensure_output_directory(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def file_exists(self, filepath):
        return Path(filepath).exists()
    
    def save_dataframe(self, df, filepath):
        df.to_csv(filepath, index=False)
        self.downloaded += 1
        logger.info(f"Saved {filepath.name}")
        

    def skip_file(self, filepath):
        self.skipped += 1
        logger.info(f"Skipped {filepath}")

    def record_failure(self, message):
        self.failed += 1
        logger.error(message)

    def summary(self):
        logger.info("=" * 50)
        logger.info("Download Summary")
        logger.info("=" * 50)
        logger.info(f"Downloaded : {self.downloaded}")
        logger.info(f"Skipped     : {self.skipped}")
        logger.info(f"Failed      : {self.failed}")