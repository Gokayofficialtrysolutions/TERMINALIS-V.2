import uuid
import datetime

class AgentManager:
    def __init__(self, config):
        self.config = config
        self.agents = []
        self.max_agents = config.get('system', {}).get('max_agents', 10)
    
    def create_agent(self, name, agent_type="generic", capabilities=None):
        """Create a new AI agent"""
        if len(self.agents) >= self.max_agents:
            raise Exception(f"Maximum number of agents ({self.max_agents}) reached")
        
        agent = {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": agent_type,
            "capabilities": capabilities or [],
            "status": "inactive",
            "created_at": datetime.datetime.now(),
            "last_active": None,
            "tasks_completed": 0
        }
        
        self.agents.append(agent)
        return agent
    
    def list_agents(self):
        """List all agents"""
        return self.agents
    
    def get_agent(self, agent_id):
        """Get agent by ID"""
        for agent in self.agents:
            if agent['id'] == agent_id:
                return agent
        return None
    
    def activate_agent(self, agent_id):
        """Activate an agent"""
        agent = self.get_agent(agent_id)
        if agent:
            agent['status'] = 'active'
            agent['last_active'] = datetime.datetime.now()
            return True
        return False
    
    def deactivate_agent(self, agent_id):
        """Deactivate an agent"""
        agent = self.get_agent(agent_id)
        if agent:
            agent['status'] = 'inactive'
            return True
        return False
    
    def remove_agent(self, agent_id):
        """Remove an agent"""
        self.agents = [agent for agent in self.agents if agent['id'] != agent_id]
    
    def get_active_agents(self):
        """Get all active agents"""
        return [agent for agent in self.agents if agent['status'] == 'active']
    
    def assign_task(self, agent_id, task):
        """Assign a task to an agent"""
        agent = self.get_agent(agent_id)
        if agent and agent['status'] == 'active':
            agent['tasks_completed'] += 1
            agent['last_active'] = datetime.datetime.now()
            # Task execution logic would go here
            return True
        return False
