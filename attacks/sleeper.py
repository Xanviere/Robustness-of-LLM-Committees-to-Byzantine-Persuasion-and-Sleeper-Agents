class SleeperAgentTracker:
    def __init__(self, K: int = 5):
        """
        K: Number of consecutive tasks to operate honestly before activating payload.
        Task index t is 1-indexed.
        """
        self.K = K
        self.current_task_index = 0
        self.activation_history = []

    def advance_task(self) -> bool:
        self.current_task_index += 1
        activated = self.is_activated()
        self.activation_history.append((self.current_task_index, activated))
        return activated

    def is_activated(self) -> bool:
        return self.current_task_index > self.K

    def get_task_index(self) -> int:
        return self.current_task_index

    def reset(self):
        self.current_task_index = 0
        self.activation_history.clear()
