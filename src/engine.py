import logging
import time

class Engine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Engine initializing...")
        self.running = False
        self.resources = {}
        self.components = {}
        self.start_time = time.time()
        self._init_subsystems()
        self.logger.info("Engine initialized successfully")
    
    def _init_subsystems(self):
        self.logger.info("Initializing engine subsystems...")
        self.components["state_manager"] = {}
        self.logger.info("Engine subsystems initialized")
    
    def update(self, delta_time):
        pass
    
    def get_uptime(self):
        return time.time() - self.start_time
    
    def register_component(self, name, component):
        self.logger.info(f"Registering component: {name}")
        self.components[name] = component
    
    def get_component(self, name):
        return self.components.get(name)
    
    def shutdown(self):
        self.logger.info("Engine shutting down...")
        self.resources.clear()
        self.components.clear()
        self.running = False
        self.logger.info("Engine shutdown complete")