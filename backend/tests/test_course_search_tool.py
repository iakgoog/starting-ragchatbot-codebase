"""
Unit tests for CourseSearchTool functionality.
Tests tool definition validation, search execution, and result formatting.
"""

from unittest.mock import Mock

import pytest
from conftest import create_mock_search_results
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolDefinition:
    """Test CourseSearchTool definition and schema validation"""

    def test_tool_definition_structure(self, course_search_tool):
        """Test that tool definition has correct structure"""
        definition = course_search_tool.get_tool_definition()

        # Check top-level structure
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition

        # Check name and description
        assert definition["name"] == "search_course_content"
        assert isinstance(definition["description"], str)
        assert len(definition["description"]) > 10

    def test_input_schema_validation(self, course_search_tool):
        """Test that input schema matches Anthropic requirements"""
        definition = course_search_tool.get_tool_definition()
        schema = definition["input_schema"]

        # Check schema structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Check required fields
        assert "query" in schema["required"]
        assert len(schema["required"]) == 1  # Only query is required

        # Check properties
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Check query property
        query_prop = properties["query"]
        assert query_prop["type"] == "string"
        assert "description" in query_prop

        # Check course_name property
        course_prop = properties["course_name"]
        assert course_prop["type"] == "string"
        assert "description" in course_prop

        # Check lesson_number property
        lesson_prop = properties["lesson_number"]
        assert lesson_prop["type"] == "integer"
        assert "description" in lesson_prop

    def test_tool_manager_registration(self, mock_vector_store):
        """Test that tool can be registered with ToolManager"""
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()

        # Should register without error
        manager.register_tool(tool)

        # Should be in tools dictionary
        assert "search_course_content" in manager.tools

        # Should appear in tool definitions
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"


class TestSearchExecution:
    """Test search execution with various scenarios"""

    def test_successful_search_basic(self, course_search_tool, mock_vector_store):
        """Test basic successful search execution"""
        # Setup mock return
        mock_results = create_mock_search_results(
            documents=["This is test content about machine learning"],
            course_titles=["AI Fundamentals"],
            lesson_numbers=[1],
        )
        mock_vector_store.search.return_value = mock_results

        # Execute search
        result = course_search_tool.execute(query="machine learning")

        # Verify call was made correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

        # Verify result format
        assert isinstance(result, str)
        assert "AI Fundamentals" in result
        assert "Lesson 1" in result
        assert "machine learning" in result

    def test_search_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test search with course name filter"""
        mock_results = create_mock_search_results(
            documents=["Course specific content"],
            course_titles=["Specific Course"],
            lesson_numbers=[2],
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(
            query="test query", course_name="Specific Course"
        )

        # Verify call includes course filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Specific Course", lesson_number=None
        )

        assert "Specific Course" in result

    def test_search_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        mock_results = create_mock_search_results(
            documents=["Lesson specific content"],
            course_titles=["Test Course"],
            lesson_numbers=[3],
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(query="test query", lesson_number=3)

        # Verify call includes lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=3
        )

        assert "Lesson 3" in result

    def test_search_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test search with both course and lesson filters"""
        mock_results = create_mock_search_results(
            documents=["Filtered content"],
            course_titles=["Filtered Course"],
            lesson_numbers=[5],
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(
            query="test query", course_name="Filtered Course", lesson_number=5
        )

        # Verify call includes both filters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Filtered Course", lesson_number=5
        )

        assert "Filtered Course" in result
        assert "Lesson 5" in result

    def test_empty_search_results(
        self, course_search_tool, mock_vector_store, mock_empty_search_results
    ):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = mock_empty_search_results

        result = course_search_tool.execute(query="nonexistent content")

        assert "No relevant content found" in result
        assert isinstance(result, str)

    def test_empty_results_with_filters(
        self, course_search_tool, mock_vector_store, mock_empty_search_results
    ):
        """Test empty results message includes filter information"""
        mock_vector_store.search.return_value = mock_empty_search_results

        # Test with course filter
        result = course_search_tool.execute(
            query="test", course_name="Nonexistent Course"
        )
        assert "No relevant content found in course 'Nonexistent Course'" in result

        # Test with lesson filter
        result = course_search_tool.execute(query="test", lesson_number=99)
        assert "No relevant content found in lesson 99" in result

        # Test with both filters
        result = course_search_tool.execute(
            query="test", course_name="Test Course", lesson_number=99
        )
        assert "in course 'Test Course'" in result
        assert "in lesson 99" in result

    def test_vector_store_error_handling(
        self, course_search_tool, mock_vector_store, mock_error_search_results
    ):
        """Test handling of vector store errors"""
        mock_vector_store.search.return_value = mock_error_search_results

        result = course_search_tool.execute(query="test query")

        # Should return the error message from SearchResults
        assert result == "Vector store connection failed"


class TestResultFormatting:
    """Test result formatting and source tracking"""

    def test_single_result_formatting(self, course_search_tool, mock_vector_store):
        """Test formatting of single search result"""
        mock_results = create_mock_search_results(
            documents=["This is test content"],
            course_titles=["Test Course"],
            lesson_numbers=[1],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = course_search_tool.execute(query="test")

        # Check format
        expected_header = "[Test Course - Lesson 1]"
        assert expected_header in result
        assert "This is test content" in result

    def test_multiple_results_formatting(self, course_search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = create_mock_search_results(
            documents=["First result content", "Second result content"],
            course_titles=["Course A", "Course B"],
            lesson_numbers=[1, 2],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        result = course_search_tool.execute(query="test")

        # Check both results are included
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First result content" in result
        assert "Second result content" in result

        # Check results are separated
        assert "\n\n" in result

    def test_source_tracking(self, course_search_tool, mock_vector_store):
        """Test that sources are tracked for UI display"""
        mock_results = create_mock_search_results(
            documents=["Test content"],
            course_titles=["Source Course"],
            lesson_numbers=[3],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"

        # Execute search
        course_search_tool.execute(query="test")

        # Check sources were tracked
        sources = course_search_tool.last_sources
        assert len(sources) == 1

        source = sources[0]
        assert source["text"] == "Source Course - Lesson 3"
        assert source["link"] == "https://example.com/lesson3"

    def test_source_tracking_multiple_results(
        self, course_search_tool, mock_vector_store
    ):
        """Test source tracking with multiple results"""
        mock_results = create_mock_search_results(
            documents=["Content 1", "Content 2"],
            course_titles=["Course 1", "Course 2"],
            lesson_numbers=[1, 2],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/course1/lesson1",
            "https://example.com/course2/lesson2",
        ]

        course_search_tool.execute(query="test")

        sources = course_search_tool.last_sources
        assert len(sources) == 2

        assert sources[0]["text"] == "Course 1 - Lesson 1"
        assert sources[0]["link"] == "https://example.com/course1/lesson1"

        assert sources[1]["text"] == "Course 2 - Lesson 2"
        assert sources[1]["link"] == "https://example.com/course2/lesson2"

    def test_missing_metadata_handling(self, course_search_tool, mock_vector_store):
        """Test handling of missing or malformed metadata"""
        # Create results with missing metadata
        mock_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.5],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(query="test")

        # Should handle gracefully
        assert "[unknown]" in result
        assert "Content with missing metadata" in result

    def test_course_without_lesson_number(self, course_search_tool, mock_vector_store):
        """Test formatting when lesson number is missing"""
        mock_results = SearchResults(
            documents=["Course content without lesson"],
            metadata=[{"course_title": "Course Only"}],  # No lesson_number
            distances=[0.5],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(query="test")

        # Should show course without lesson number
        assert "[Course Only]" in result
        assert "Course content without lesson" in result

        # Check source tracking without lesson
        sources = course_search_tool.last_sources
        assert len(sources) == 1
        assert sources[0]["text"] == "Course Only"


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    def test_none_query_handling(self, course_search_tool, mock_vector_store):
        """Test handling of None query parameter"""
        # This should cause an error in the vector store call
        mock_vector_store.search.side_effect = TypeError("Query cannot be None")

        with pytest.raises(TypeError):
            course_search_tool.execute(query=None)

    def test_empty_query_handling(self, course_search_tool, mock_vector_store):
        """Test handling of empty query"""
        mock_results = create_mock_search_results(
            documents=[], course_titles=[], lesson_numbers=[]
        )
        mock_vector_store.search.return_value = mock_results

        result = course_search_tool.execute(query="")

        # Should handle empty query gracefully
        assert "No relevant content found" in result

    def test_invalid_lesson_number(self, course_search_tool, mock_vector_store):
        """Test handling of invalid lesson numbers"""
        mock_results = create_mock_search_results(
            documents=[], course_titles=[], lesson_numbers=[]
        )
        mock_vector_store.search.return_value = mock_results

        # Negative lesson number
        result = course_search_tool.execute(query="test", lesson_number=-1)
        assert "No relevant content found in lesson -1" in result

        # Zero lesson number
        result = course_search_tool.execute(query="test", lesson_number=0)
        assert "No relevant content found in lesson 0" in result

    def test_tool_manager_execution(self, tool_manager):
        """Test tool execution through ToolManager"""
        # Execute tool through manager
        result = tool_manager.execute_tool(
            "search_course_content", query="test query", course_name="Test Course"
        )

        # Should return formatted results (using mock data from conftest)
        assert isinstance(result, str)
        assert "Test Course" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
