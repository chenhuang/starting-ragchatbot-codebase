import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator

class TestAIGenerator:
    """Test cases for AIGenerator functionality"""
    
    def test_init_with_config(self):
        """Test AIGenerator initialization with proper config"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
            
            mock_anthropic.assert_called_once_with(api_key="test-api-key")
            assert ai_gen.model == "claude-sonnet-4-20250514"
            assert ai_gen.max_tool_rounds == 2  # Default value
            assert ai_gen.base_params["model"] == "claude-sonnet-4-20250514"
            assert ai_gen.base_params["temperature"] == 0
            assert ai_gen.base_params["max_tokens"] == 800
            
    def test_init_with_custom_max_tool_rounds(self):
        """Test AIGenerator initialization with custom max_tool_rounds"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514", max_tool_rounds=3)
            
            mock_anthropic.assert_called_once_with(api_key="test-api-key")
            assert ai_gen.max_tool_rounds == 3
    
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test generate_response() without tools (direct response)"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            response = ai_gen.generate_response("What is Python?")
            
            # Verify API call
            mock_anthropic_client.messages.create.assert_called_once()
            call_args = mock_anthropic_client.messages.create.call_args[1]
            
            assert call_args["model"] == "claude-sonnet-4-20250514"
            assert call_args["temperature"] == 0
            assert call_args["max_tokens"] == 800
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "What is Python?"
            assert "tools" not in call_args  # No tools provided
            
            assert response == "This is a test response"
    
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test generate_response() with conversation history"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            history = "User: Hello\nAssistant: Hi there!"
            response = ai_gen.generate_response("What is Python?", conversation_history=history)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert history in call_args["system"]
    
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test generate_response() with tools available but not used"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            tools = [{"name": "search_course_content", "description": "Search courses"}]
            response = ai_gen.generate_response("General question", tools=tools)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tools"] == tools
            assert call_args["tool_choice"] == {"type": "auto"}
            
            assert response == "This is a test response"
    
    def test_generate_response_with_tool_use(self, mock_anthropic_tool_use_client):
        """Test generate_response() when Claude decides to use a tool"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Python basics"
        
        # Mock the final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Based on the search, Python is a programming language")]
        mock_anthropic_tool_use_client.messages.create.side_effect = [
            mock_anthropic_tool_use_client.messages.create.return_value,  # First call with tool use
            final_response  # Second call with final response
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_tool_use_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            tools = [{"name": "search_course_content", "description": "Search courses"}]
            response = ai_gen.generate_response("What is Python?", tools=tools, tool_manager=mock_tool_manager)
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", 
                query="Python basics"
            )
            
            # Verify final response
            assert response == "Based on the search, Python is a programming language"
    
    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution() with single tool call"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            # Mock initial response with tool use
            initial_response = Mock()
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.id = "tool_123"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "Python"}
            initial_response.content = [mock_tool_block]
            
            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Mock final API response
            final_response = Mock()
            final_response.content = [Mock(text="Final answer")]
            mock_anthropic.return_value.messages.create.return_value = final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "What is Python?"}],
                "system": "System prompt",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0,
                "max_tokens": 800
            }
            
            result = ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            # Verify tool execution
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", 
                query="Python"
            )
            
            # Verify final API call structure
            final_call_args = mock_anthropic.return_value.messages.create.call_args[1]
            assert len(final_call_args["messages"]) == 3  # Original + Assistant + Tool results
            assert final_call_args["messages"][1]["role"] == "assistant"
            assert final_call_args["messages"][2]["role"] == "user"
            assert final_call_args["messages"][2]["content"][0]["type"] == "tool_result"
            assert final_call_args["messages"][2]["content"][0]["tool_use_id"] == "tool_123"
            assert final_call_args["messages"][2]["content"][0]["content"] == "Tool execution result"
            
            assert result == "Final answer"
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution() with multiple tool calls"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            # Mock initial response with multiple tool uses
            initial_response = Mock()
            mock_tool_block1 = Mock()
            mock_tool_block1.type = "tool_use"
            mock_tool_block1.id = "tool_1"
            mock_tool_block1.name = "search_course_content"
            mock_tool_block1.input = {"query": "Python"}
            
            mock_tool_block2 = Mock()
            mock_tool_block2.type = "tool_use"
            mock_tool_block2.id = "tool_2"
            mock_tool_block2.name = "get_course_outline"
            mock_tool_block2.input = {"course_title": "Python Basics"}
            
            initial_response.content = [mock_tool_block1, mock_tool_block2]
            
            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "Search result",
                "Outline result"
            ]
            
            # Mock final API response
            final_response = Mock()
            final_response.content = [Mock(text="Combined answer")]
            mock_anthropic.return_value.messages.create.return_value = final_response
            
            base_params = {
                "messages": [{"role": "user", "content": "Query"}],
                "system": "System prompt"
            }
            
            result = ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Python Basics")
            
            # Verify tool results structure
            final_call_args = mock_anthropic.return_value.messages.create.call_args[1]
            tool_results = final_call_args["messages"][2]["content"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_use_id"] == "tool_1"
            assert tool_results[1]["tool_use_id"] == "tool_2"
            
            assert result == "Combined answer"
    
    def test_tool_execution_without_tool_manager(self, mock_anthropic_tool_use_client):
        """Test that when Claude tries to use tools but no tool_manager is available, 
        an appropriate error is returned"""
        
        # Mock the second API call (after tool error)
        final_response = Mock()
        final_response.content = [Mock(text="I apologize, but I cannot access the search tools right now.")]
        final_response.stop_reason = "end_turn"
        
        mock_anthropic_tool_use_client.messages.create.side_effect = [
            mock_anthropic_tool_use_client.messages.create.return_value,  # First call with tool_use
            final_response  # Second call after tool error
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_tool_use_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            response = ai_gen.generate_response("What is Python?", tools=[{"name": "test"}])
            
            # Should return final response that acknowledges the tool error
            assert "cannot access" in response or "error" in response.lower()
            
            # Verify two API calls were made
            assert mock_anthropic_tool_use_client.messages.create.call_count == 2
    
    def test_system_prompt_content(self):
        """Test that the system prompt contains expected tool instructions"""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in AIGenerator.SYSTEM_PROMPT
        assert "Content queries" in AIGenerator.SYSTEM_PROMPT
        assert "Outline queries" in AIGenerator.SYSTEM_PROMPT
        # Check for sequential tool calling support
        assert "Multi-step queries" in AIGenerator.SYSTEM_PROMPT
        assert "Sequential reasoning" in AIGenerator.SYSTEM_PROMPT
        # Ensure old limitation is removed
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
    
    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling with two rounds"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 4 is about Python Functions",
            "Found course: Advanced Python covering Functions topic"
        ]
        
        # Create mock responses for sequential calls
        # Round 1: Tool use (get outline)
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.id = "tool_1"
        mock_tool1.name = "get_course_outline"
        mock_tool1.input = {"course_title": "Python Basics"}
        round1_response.content = [mock_tool1]
        
        # Round 2: Tool use (search for similar course)
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.id = "tool_2"
        mock_tool2.name = "search_course_content"
        mock_tool2.input = {"query": "Functions topic"}
        round2_response.content = [mock_tool2]
        
        # Final response: No tool use
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Found Advanced Python course that covers Functions like lesson 4")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            round1_response,  # Round 1 API call
            round2_response,  # Round 2 API call  
            final_response    # Final API call
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search content"}
            ]
            response = ai_gen.generate_response(
                "Search for a course that discusses the same topic as lesson 4 of Python Basics",
                tools=tools, 
                tool_manager=mock_tool_manager
            )
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Python Basics")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Functions topic")
            
            # Verify 3 API calls were made (2 tool rounds + 1 final)
            assert mock_client.messages.create.call_count == 3
            
            # Verify final response
            assert response == "Found Advanced Python course that covers Functions like lesson 4"
    
    def test_sequential_tool_calling_max_rounds_reached(self):
        """Test that tool calling stops after max_tool_rounds"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create responses that always want to use tools
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            Mock(type="tool_use", id="tool_1", name="search_course_content", input={"query": "test"})
        ]
        
        # Final response when tools are removed
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Final response after max rounds")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            tool_response,    # Round 1
            tool_response,    # Round 2
            final_response    # Final call without tools
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514", max_tool_rounds=2)
            
            tools = [{"name": "search_course_content", "description": "Search"}]
            response = ai_gen.generate_response(
                "Complex query requiring multiple searches",
                tools=tools, 
                tool_manager=mock_tool_manager
            )
            
            # Verify max rounds were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            
            # Verify 3 API calls: 2 tool rounds + 1 final without tools
            assert mock_client.messages.create.call_count == 3
            
            # Check that final call had no tools
            final_call_args = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in final_call_args
            
            assert response == "Final response after max rounds"
    
    def test_sequential_tool_calling_stops_on_no_tool_use(self):
        """Test that tool calling stops when Claude doesn't use tools"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Round 1: Tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [
            Mock(type="tool_use", id="tool_1", name="search_course_content", input={"query": "test"})
        ]
        
        # Round 2: No tool use (Claude decides it has enough info)
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        round2_response.content = [Mock(text="Based on the search, here's my answer")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            round1_response,   # Round 1 with tool use
            round2_response    # Round 2 without tool use - should stop here
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            tools = [{"name": "search_course_content", "description": "Search"}]
            response = ai_gen.generate_response(
                "Query that needs one search",
                tools=tools, 
                tool_manager=mock_tool_manager
            )
            
            # Verify only one tool was executed
            assert mock_tool_manager.execute_tool.call_count == 1
            
            # Verify only 2 API calls (no final call needed)
            assert mock_client.messages.create.call_count == 2
            
            assert response == "Based on the search, here's my answer"
    
    def test_sequential_tool_calling_tool_execution_error(self):
        """Test error handling when tool execution fails"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        # Configure tool_response mock properly
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        
        # Create tool block that will trigger tool execution (which will fail)  
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_1"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        
        # Configure the mock so hasattr(mock_tool, 'text') returns False
        # This prevents the fallback from trying to access .text on the tool block
        del mock_tool.text  # Remove the .text attribute so hasattr returns False
        
        tool_response.content = [mock_tool]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = tool_response
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            tools = [{"name": "search_course_content", "description": "Search"}]
            response = ai_gen.generate_response(
                "Query that will fail",
                tools=tools, 
                tool_manager=mock_tool_manager
            )
            
            # Should return hardcoded fallback response when tool execution fails
            assert response == "I encountered an error while processing your request. Please try again."
            
            # Verify tool execution was attempted
            assert mock_tool_manager.execute_tool.call_count == 1