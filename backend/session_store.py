_sessions = {}

def get_session(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = {}
    return _sessions[session_id]

def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
