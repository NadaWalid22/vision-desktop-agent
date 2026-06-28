"""Desktop automation layer."""
from src.automation.desktop_controller import DesktopControlError, DesktopController
from src.automation.notepad_workflow import NotepadWorkflow, NotepadWorkflowError

__all__ = [
    "DesktopController",
    "DesktopControlError",
    "NotepadWorkflow",
    "NotepadWorkflowError",
]
