# sentry/api/routes/chat.py
"""
Chat history endpoints
"""

from fastapi import APIRouter, Depends, status

from ..schemas import ChatHistoryResponse, ChatMessageResponse
from ..auth import verify_api_key
from ...core.database import db

router = APIRouter(prefix="/chat", tags=["Chat"], dependencies=[Depends(verify_api_key)])


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(limit: int = 50):
    """
    Get recent chat history
    
    Args:
        limit: Maximum number of messages to return (default 50)
    """
    messages = db.get_chat_history(limit=limit)
    
    return ChatHistoryResponse(
        messages=[
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                sources=msg.sources
            )
            for msg in messages
        ],
        total=len(messages)
    )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history():
    """Clear all chat history"""
    db.clear_chat_history()
