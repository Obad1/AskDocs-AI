from pydantic import BaseModel, Field
from typing import List, Optional

class QuizQuestion(BaseModel):
    question_text: str = Field(..., description="The quiz question generated from the text")
    choices: List[str] = Field(..., description="Exactly 4 choices; raw text, no list markers")
    answer: str = Field(..., description="The exact string matching the correct choice")
    explanation: str = Field(..., description="Reasoning behind the correct answer")

class QuizModel(BaseModel):
    quiz_title: str = Field(..., description="A concise title reflecting the document chapter")
    questions: List[QuizQuestion] = Field(..., description="List of generated quiz questions")

class FlashcardItem(BaseModel):
    front: str = Field(..., description="Key concept or question")
    back: str = Field(..., description="Brief explanation or answer")

class FlashcardDeck(BaseModel):
    deck_name: str = Field(..., description="Name of the generated deck")
    cards: List[FlashcardItem]

class KnowledgeGraphNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Display label of the entity")
    type: str = Field(..., description="Entity type: Concept, Definition, Algorithm, Person, Location")
    color: str = Field(default="#3b82f6", description="Node color based on type")

class KnowledgeGraphEdge(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    label: str = Field(default="", description="Relationship label")

class KnowledgeGraphModel(BaseModel):
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]
