import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ConversationState:
    """Tracks conversation state across tool calling rounds"""
    initial_query: str
    system_content: str
    tools: Optional[List] = None
    tool_manager: Optional[Any] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    last_response: Optional[Any] = None
    
    def __post_init__(self):
        if not self.messages:
            self.messages = [{"role": "user", "content": self.initial_query}]

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **Search Tool (search_course_content)**: For finding specific course content and detailed educational materials
2. **Outline Tool (get_course_outline)**: For retrieving complete course structures including:
   - Course title and link
   - Complete lesson list with numbers and titles
   - Total lesson count

Tool Usage Guidelines:
- **Outline queries**: Use get_course_outline for questions about course structure, lesson lists, or course overviews
- **Content queries**: Use search_course_content for specific educational content within courses
- **Multi-step queries**: You can make multiple tool calls to gather comprehensive information
- **Sequential reasoning**: Use tool results to inform subsequent searches
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Complex questions**: Break down into multiple searches as needed
- **Comparisons**: Search for each item separately then compare
- **Multi-part questions**: Address each part with appropriate tool calls
- **Course outline questions**: Use get_course_outline tool first, then provide formatted response with course title, course link, and complete lesson breakdown
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Now supports sequential tool calling up to max_tool_rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation state
        conversation_state = ConversationState(
            initial_query=query,
            system_content=system_content,
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Execute the tool calling loop
        return self._execute_tool_calling_loop(conversation_state)
    
    def _execute_tool_calling_loop(self, state: ConversationState) -> str:
        """
        Execute the main tool calling loop with up to max_tool_rounds iterations.
        
        Termination conditions:
        1. Reached max_tool_rounds
        2. Claude's response contains no tool_use blocks
        3. Tool execution fails
        
        Returns:
            Final response text
        """
        
        while state.current_round < self.max_tool_rounds:
            state.current_round += 1
            
            # Make API call with tools available
            try:
                response = self._make_api_call_with_tools(state)
                state.last_response = response
                
                # Check termination condition: no tool use
                if response.stop_reason != "tool_use":
                    # Handle response content safely
                    if not response.content:
                        return "I was unable to generate a response. Please try again."
                    
                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == "text":
                            return content_block.text
                        elif hasattr(content_block, 'text'):
                            return content_block.text
                    
                    return "I was unable to generate a complete response. Please try again."
                
                # Execute tools and update conversation state
                if not self._execute_tools_and_update_state(response, state):
                    # Tool execution failed - return last valid response
                    return self._get_fallback_response(state)
                    
            except Exception as e:
                # API call failed - return error or fallback
                return self._handle_api_error(e, state)
        
        # Max rounds reached - make final call without tools
        return self._make_final_response(state)
    
    def _make_api_call_with_tools(self, state: ConversationState):
        """Make API call with tools available"""
        api_params = {
            **self.base_params,
            "messages": state.messages.copy(),
            "system": state.system_content
        }
        
        # Always include tools if available (this is the key change!)
        if state.tools:
            api_params["tools"] = state.tools
            api_params["tool_choice"] = {"type": "auto"}
        
        return self.client.messages.create(**api_params)
    
    def _execute_tools_and_update_state(self, response, state: ConversationState) -> bool:
        """
        Execute all tool calls and update conversation state.
        
        Returns:
            True if successful, False if any tool execution failed
        """
        # Add Claude's response to conversation
        state.messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls
        tool_results = []
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                # Check if tool_manager is available
                if not state.tool_manager:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": "Error: Tool execution not available - no tool manager provided"
                    })
                    continue
                    
                try:
                    tool_result = state.tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    # Tool execution failed
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}",
                        "is_error": True
                    })
                    return False
        
        # Add tool results to conversation
        if tool_results:
            state.messages.append({"role": "user", "content": tool_results})
        
        return True
    
    def _make_final_response(self, state: ConversationState) -> str:
        """
        Make final API call without tools when max rounds reached.
        This ensures Claude provides a final answer based on all tool results.
        """
        final_params = {
            **self.base_params,
            "messages": state.messages,
            "system": state.system_content
            # Deliberately no tools here for final synthesis
        }
        
        final_response = self.client.messages.create(**final_params)
        
        # Handle empty response content
        if not final_response.content:
            return "I was unable to generate a response. Please try again."
        
        # Find text content
        for content_block in final_response.content:
            if hasattr(content_block, 'type') and content_block.type == "text":
                return content_block.text
            elif hasattr(content_block, 'text'):
                return content_block.text
        
        # Fallback if no text content found
        return "I was unable to generate a complete response. Please try again."
    
    def _get_fallback_response(self, state: ConversationState) -> str:
        """Get fallback response when tool execution fails"""
        if state.last_response and state.last_response.content:
            # Try to extract any text content from the last response
            for content_block in state.last_response.content:
                if hasattr(content_block, 'type') and content_block.type == "text":
                    return content_block.text
                elif hasattr(content_block, 'text'):
                    return content_block.text
        
        return "I encountered an error while processing your request. Please try again."
    
    def _handle_api_error(self, error: Exception, state: ConversationState) -> str:
        """Handle API call errors"""
        return f"I'm experiencing technical difficulties. Please try your request again."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text