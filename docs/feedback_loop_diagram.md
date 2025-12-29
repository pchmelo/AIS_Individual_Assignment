# Human-in-the-Loop Feedback Architecture Diagram

```mermaid
graph TD
    Start([Pipeline Stage N]) --> Execute[Execute Stage Analysis]
    Execute --> Present[Present Results to User]
    Present --> Feedback{User Provides<br/>Feedback?}
    
    Feedback -->|Yes| FeedbackAgent[Feedback Interpretation Agent]
    Feedback -->|No - Approve| NextStage([Continue to Stage N+1])
    
    FeedbackAgent --> Interpret[Interpret Natural Language Input]
    Interpret --> Decide{Decision Type?}
    
    Decide -->|Rerun Stage| Adjust[Adjust Parameters<br/>for Current Stage]
    Decide -->|Change Tool| Switch[Invoke Alternative Tool]
    Decide -->|Skip Stage| Skip([Jump to Stage N+1])
    Decide -->|Need More Info| Clarify[Request Clarification<br/>from User]
    
    Adjust --> Log1[Log Modification]
    Switch --> Log1
    Clarify --> Present
    
    Log1 --> Rerun[Re-execute Stage N]
    Rerun --> Present
    
    Skip --> Log2[Log Skip Decision]
    Log2 --> NextStage
    
    NextStage --> Check{More Stages?}
    Check -->|Yes| Start
    Check -->|No| Report([Generate Final Report<br/>with Feedback History])
    
    style FeedbackAgent fill:#3498db,stroke:#2980b9,color:#fff
    style Feedback fill:#e74c3c,stroke:#c0392b,color:#fff
    style Decide fill:#f39c12,stroke:#d68910,color:#fff
    style Report fill:#27ae60,stroke:#229954,color:#fff
```

## Architecture Components

**Key Elements:**

1. **Pipeline Stage N**: Any stage in the 7-stage evaluation pipeline
2. **Feedback Interpretation Agent**: New AI component that parses natural language feedback
3. **Decision Types**:
   - **Rerun Stage**: Adjust parameters and re-execute
   - **Change Tool**: Switch to alternative analytical tool
   - **Skip Stage**: Jump to next stage if current is not relevant
   - **Need More Info**: Request user clarification
4. **Logging**: All feedback-driven modifications are tracked for transparency
5. **Final Report**: Includes complete history of human feedback and AI adaptations

**Benefits:**
- Iterative refinement based on domain expertise
- Flexible pipeline execution adapting to user needs
- Transparent decision trail for reproducibility
- Balance between automation and human control
