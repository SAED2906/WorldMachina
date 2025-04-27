#!/usr/bin/env python3

import sys
import logging
from engine import Engine
from window import Window

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting application...")
    
    try:
        logger.info("Initializing engine...")
        engine = Engine()
        
        logger.info("Creating window...")
        window = Window(engine)
        
        logger.info("Running application...")
        window.run()
        
        logger.info("Shutting down...")
        window.cleanup()
        engine.shutdown()
        logger.info("Application terminated successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())