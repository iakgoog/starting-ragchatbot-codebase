"""
Unit tests for AIGenerator functionality.
Tests tool registration, tool calling, and response processing.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import anthropic
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorInitialization:
    """Test AIGenerator initialization and setup"""
    
    def test_initialization_with_valid_params(self, test_config):
        """Test AIGenerator initializes correctly with valid parameters"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        assert ai_gen.model == test_config.ANTHROPIC_MODEL
        assert ai_gen.client is not None
        assert isinstance(ai_gen.client, anthropic.Anthropic)
    
    def test_base_params_setup(self, test_config):
        """Test that base parameters are set up correctly"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        expected_params = {
            "model": test_config.ANTHROPIC_MODEL,
            "temperature": 0,
            "max_tokens": 800
        }
        
        assert ai_gen.base_params == expected_params
    
    def test_system_prompt_is_defined(self):
        """Test that system prompt is properly defined"""
        assert hasattr(AIGenerator, 'SYSTEM_PROMPT')
        assert isinstance(AIGenerator.SYSTEM_PROMPT, str)
        assert len(AIGenerator.SYSTEM_PROMPT) > 100
        assert "course materials" in AIGenerator.SYSTEM_PROMPT.lower()


class TestBasicResponseGeneration:
    """Test basic response generation without tools"""
    
    def test_simple_response_generation(self, ai_generator_mock, mock_anthropic_client):
        """Test generating simple response without tools"""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a test response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Generate response
        result = ai_generator_mock.generate_response("Hello, how are you?")
        
        # Verify call was made
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        
        # Check call structure
        assert "messages" in call_args.kwargs
        assert "system" in call_args.kwargs
        assert call_args.kwargs["model"] == ai_generator_mock.model
        assert call_args.kwargs["temperature"] == 0
        assert call_args.kwargs["max_tokens"] == 800
        
        # Check message content
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how are you?"
        
        # Check result
        assert result == "This is a test response"
    
    def test_response_with_conversation_history(self, ai_generator_mock, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with context"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "User: Previous question\nAssistant: Previous answer"
        
        result = ai_generator_mock.generate_response(
            "Follow-up question",
            conversation_history=history
        )
        
        # Check that history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content
        assert result == "Response with context"


class TestToolRegistration:
    """Test tool registration and definition handling"""
    
    def test_generate_response_with_tools(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test response generation with tools available"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response using tools"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Get tool definitions from manager
        tool_definitions = tool_manager.get_tool_definitions()
        
        result = ai_generator_mock.generate_response(
            "Search for something",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify tools were passed to API
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}
        
        # Check tool definitions structure
        tools = call_args.kwargs["tools"]
        assert len(tools) >= 1
        assert tools[0]["name"] in ["search_course_content", "get_course_outline"]
        
        assert result == "Response using tools"
    
    def test_tool_definitions_format(self, tool_manager):
        """Test that tool definitions have correct format for Anthropic API"""
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) > 0
        
        for definition in definitions:
            # Check required fields
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition
            
            # Check schema structure
            schema = definition["input_schema"]
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestToolExecution:
    """Test tool calling and execution flow"""
    
    def test_tool_execution_flow(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test complete tool execution flow"""
        # Setup initial tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        # Mock tool use content block
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_call_123"
        tool_use_block.input = {"query": "test search", "course_name": "Test Course"}
        
        initial_response.content = [tool_use_block]
        
        # Setup final response after tool execution
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Here are the search results: ..."
        
        # Configure mock to return different responses for each call
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        # Execute with tools
        result = ai_generator_mock.generate_response(
            "Search for test content",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Check first call (with tools)
        first_call = mock_anthropic_client.messages.create.call_args_list[0]
        assert "tools" in first_call.kwargs
        assert "tool_choice" in first_call.kwargs
        
        # Check second call (without tools, with tool results)
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        assert "tools" not in second_call.kwargs
        assert len(second_call.kwargs["messages"]) == 3  # original + assistant + tool results
        
        # Check final result
        assert result == "Here are the search results: ..."
    
    def test_tool_result_format(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test that tool results are formatted correctly for API"""
        # Setup tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_call_456"
        tool_use_block.input = {"query": "machine learning"}
        
        initial_response.content = [tool_use_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Tool execution completed"
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        result = ai_generator_mock.generate_response(
            "Tell me about machine learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check tool result message format in second call
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        
        # Find the tool result message
        tool_result_message = messages[2]  # Should be third message
        assert tool_result_message["role"] == "user"
        
        tool_results = tool_result_message["content"]
        assert len(tool_results) == 1
        
        tool_result = tool_results[0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_call_456"
        assert "content" in tool_result
    
    def test_multiple_tool_calls(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test handling of multiple tool calls in one response"""
        # Setup response with multiple tool calls
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_call_1"
        tool_use_1.input = {"query": "first search"}
        
        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "get_course_outline"
        tool_use_2.id = "tool_call_2"
        tool_use_2.input = {"course_name": "Test Course"}
        
        initial_response.content = [tool_use_1, tool_use_2]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Multiple tools executed"
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        result = ai_generator_mock.generate_response(
            "Complex query requiring multiple tools",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check that both tools were executed
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_result_message = messages[2]
        tool_results = tool_result_message["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_call_1"
        assert tool_results[1]["tool_use_id"] == "tool_call_2"


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_anthropic_api_error(self, ai_generator_mock, mock_anthropic_client):
        """Test handling of Anthropic API errors"""
        # Setup API error - use generic exception to avoid complex APIError instantiation
        mock_anthropic_client.messages.create.side_effect = Exception("API Error occurred")
        
        with pytest.raises(Exception, match="API Error occurred"):
            ai_generator_mock.generate_response("Test query")
    
    def test_tool_execution_error(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test handling of tool execution errors"""
        # Setup tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "nonexistent_tool"  # This will cause an error
        tool_use_block.id = "tool_call_error"
        tool_use_block.input = {"query": "test"}
        
        initial_response.content = [tool_use_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Tool error handled"
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        result = ai_generator_mock.generate_response(
            "Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should complete despite tool error
        assert result == "Tool error handled"
        
        # Check that error message was passed to second API call
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_result_message = messages[2]
        tool_results = tool_result_message["content"]
        
        # Error should be in tool result content
        assert "Tool 'nonexistent_tool' not found" in tool_results[0]["content"]
    
    def test_malformed_tool_response(self, ai_generator_mock, mock_anthropic_client, tool_manager):
        """Test handling of malformed tool use responses"""
        # Setup malformed response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = []  # Empty content
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Handled malformed response"
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        result = ai_generator_mock.generate_response(
            "Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should handle gracefully
        assert result == "Handled malformed response"
    
    def test_missing_tool_manager(self, ai_generator_mock, mock_anthropic_client):
        """Test handling when tool_manager is None but tools are available"""
        # Setup tool use response but no tool manager
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Should not reach here"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # This should not attempt tool execution without manager
        result = ai_generator_mock.generate_response(
            "Test query",
            tools=[{"name": "test_tool"}],
            tool_manager=None
        )
        
        # Should return the text from the tool_use response since no manager available
        assert result == "Should not reach here"


class TestResponseTypes:
    """Test different types of responses"""
    
    def test_text_only_response(self, ai_generator_mock, mock_anthropic_client):
        """Test normal text-only response"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Pure text response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator_mock.generate_response("Simple question")
        assert result == "Pure text response"
    
    def test_stop_reason_handling(self, ai_generator_mock, mock_anthropic_client):
        """Test different stop reasons are handled correctly"""
        # Test max_tokens stop reason
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Truncated response"
        mock_response.stop_reason = "max_tokens"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator_mock.generate_response("Long question")
        assert result == "Truncated response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])