import hashlib
import types
import typing as t
from inspect import Signature

import autogen
import typing_extensions as te
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen_core.tools import FunctionTool

from composio import Action, ActionType, AppType, TagType
from composio.tools import ComposioToolSet as BaseComposioToolSet
from composio.tools.toolset import ProcessorsType
from composio.utils.shared import get_signature_format_from_schema_params


class ComposioToolSet(
    BaseComposioToolSet,
    runtime="autogen",
    description_char_limit=1024,
    action_name_char_limit=64,
):
    """
    Composio toolset for autogen framework.
    """

    def register_tools(
        self,
        caller: ConversableAgent,
        executor: ConversableAgent,
        actions: t.Optional[t.Sequence[ActionType]] = None,
        apps: t.Optional[t.Sequence[AppType]] = None,
        tags: t.Optional[t.List[TagType]] = None,
        entity_id: t.Optional[str] = None,
    ) -> None:
        """
        Register tools to the proxy agents.

        :param executor: Executor agent.
        :param caller: Caller agent.
        :param apps: List of apps to wrap
        :param actions: List of actions to wrap
        :param tags: Filter the apps by given tags
        :param entity_id: Entity ID for the function wrapper
        :param entity_id: Entity ID to use for executing function calls.
        """
        self.validate_tools(apps=apps, actions=actions, tags=tags)
        schemas = self.get_action_schemas(
            actions=actions,
            apps=apps,
            tags=tags,
            _populate_requested=True,
        )
        for schema in schemas:
            self._register_schema_to_autogen(
                schema=schema.model_dump(
                    exclude_defaults=True,
                    exclude_none=True,
                    exclude_unset=True,
                ),
                caller=caller,
                executor=executor,
                entity_id=entity_id or self.entity_id,
            )

    @te.deprecated("Use `ComposioToolSet.register_tools` instead")
    def register_actions(
        self,
        caller: ConversableAgent,
        executor: ConversableAgent,
        actions: t.Sequence[ActionType],
        entity_id: t.Optional[str] = None,
    ):
        """
        Register tools to the proxy agents.

        :param actions: List of tools to register.
        :param caller: Caller agent.
        :param executor: Executor agent.
        :param entity_id: Entity ID to use for executing function calls.
        """
        self.register_tools(
            caller=caller,
            executor=executor,
            actions=actions,
            entity_id=entity_id,
        )

    def _process_function_name_for_registration(
        self,
        input_string: str,
        max_allowed_length: int = 64,
        num_hash_char: int = 10,
    ):
        """
        Process function name for proxy registration under given character length limitation.
        """
        hash_hex = hashlib.sha256(input_string.encode(encoding="utf-8")).hexdigest()
        hash_chars_to_attach = hash_hex[:10]
        num_input_str_char = max_allowed_length - (num_hash_char + 1)
        input_str_to_attach = input_string[-num_input_str_char:]
        processed_name = input_str_to_attach + "_" + hash_chars_to_attach
        return processed_name

    def _register_schema_to_autogen(
        self,
        schema: t.Dict,
        caller: ConversableAgent,
        executor: ConversableAgent,
        entity_id: t.Optional[str] = None,
    ) -> None:
        """
        Register a schema to the Autogen registry.

        Args:
            schema (dict[str, any]): The action schema to be registered.
            caller (ConversableAgent): The agent responsible for initiating the tool registration.
            executor (ConversableAgent): The agent responsible for executing the registered tools.
            entity_id (str, optional): The identifier of the entity for which the action is executed. Defaults to None.
        """
        name = schema["name"]
        appName = schema["appName"]
        description = schema["description"]

        def execute_action(**kwargs: t.Any) -> t.Dict:
            """Placeholder function for executing action."""
            return self.execute_action(
                action=Action(value=name),
                params=kwargs,
                entity_id=entity_id or self.entity_id,
                _check_requested_actions=True,
            )

        function = types.FunctionType(
            code=execute_action.__code__,
            globals=globals(),
            name=self._process_function_name_for_registration(
                input_string=name,
            ),
            closure=execute_action.__closure__,
        )
        params = get_signature_format_from_schema_params(
            schema_params=schema["parameters"],
        )
        setattr(function, "__signature__", Signature(parameters=params))
        setattr(
            function,
            "__annotations__",
            {p.name: p.annotation for p in params} | {"return": t.Dict[str, t.Any]},
        )
        function.__doc__ = (
            description if description else f"Action {name} from {appName}"
        )
        autogen.agentchat.register_function(
            function,
            caller=caller,
            executor=executor,
            name=self._process_function_name_for_registration(
                input_string=name,
            ),
            description=description if description else f"Action {name} from {appName}",
        )

    def _wrap_tool(
        self,
        schema: t.Dict[str, t.Any],
        entity_id: t.Optional[str] = None,
        skip_default: bool = False,
    ) -> FunctionTool:
        """
        Wraps a composio action as an Autogen FunctionTool.

        Args:
            schema: The action schema to wrap
            entity_id: Optional entity ID for executing function calls

        Returns:
            FunctionTool: Wrapped function as an Autogen FunctionTool
        """
        name = schema["name"]
        description = schema["description"] or f"Action {name} from {schema['appName']}"

        # Generate a new function dynamically with the processed schema
        # This ensures FunctionTool generates the correct schema with processed descriptions
        function = self._create_function_with_processed_schema(
            name=name,
            schema=schema,
            entity_id=entity_id,
            skip_default=skip_default,
            description=description,
        )

        # Create a custom FunctionTool that uses our processed schema
        tool = self._create_custom_function_tool(
            func=function,
            description=description,
            name=self._process_function_name_for_registration(input_string=name),
            processed_schema=schema,
        )
        
        return tool

    def _create_function_with_processed_schema(
        self,
        name: str,
        schema: t.Dict[str, t.Any],
        entity_id: t.Optional[str],
        skip_default: bool,
        description: str,
    ) -> t.Callable:
        """
        Create a function dynamically with the processed schema properties.
        This ensures that FunctionTool generates the correct schema with processed descriptions.
        """
        from pydantic import Field
        import inspect
        
        # Get the processed properties from the schema
        properties = schema["parameters"].get("properties", {})
        required_params = schema["parameters"].get("required", [])
        
        # Build the function signature dynamically
        params = []
        annotations = {}
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_default = param_info.get("default")
            param_description = param_info.get("description", param_name)
            is_required = param_name in required_params
            
            # Map JSON schema types to Python types
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            python_type = type_mapping.get(param_type, str)
            
            # Create parameter with Field annotation for description
            if is_required and not skip_default:
                default_value = Field(..., description=param_description)
            elif param_default is not None:
                default_value = Field(default=param_default, description=param_description)
            else:
                default_value = Field(default=None, description=param_description)
            
            # Add to signature
            if is_required and skip_default:
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=python_type,
                )
            else:
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=python_type,
                    default=default_value,
                )
            
            params.append(param)
            annotations[param_name] = python_type
        
        # Add return type annotation
        annotations["return"] = t.Dict[str, t.Any]
        
        # Create the function signature
        sig = inspect.Signature(params)
        
        # Create the actual function
        def dynamic_function(**kwargs: t.Any) -> t.Dict:
            """Dynamically generated function for executing action."""
            return self.execute_action(
                action=Action(value=name),
                params=kwargs,
                entity_id=entity_id or self.entity_id,
                _check_requested_actions=True,
            )
        
        # Set the signature, annotations, and docstring
        dynamic_function.__signature__ = sig
        dynamic_function.__annotations__ = annotations
        dynamic_function.__name__ = self._process_function_name_for_registration(input_string=name)
        dynamic_function.__doc__ = description
        
        return dynamic_function

    def _create_custom_function_tool(
        self,
        func: t.Callable,
        description: str,
        name: str,
        processed_schema: t.Dict[str, t.Any],
    ) -> FunctionTool:
        """
        Create a custom FunctionTool that uses our processed schema instead of auto-generating it.
        """
        # Create a subclass of FunctionTool that overrides the schema property
        class CustomFunctionTool(FunctionTool):
            def __init__(self, func, description, name, processed_schema):
                super().__init__(func=func, description=description, name=name)
                self._processed_schema = {
                    "name": name,
                    "description": description,
                    "parameters": processed_schema["parameters"],
                    "strict": False
                }
            
            @property
            def schema(self):
                return self._processed_schema
        
        return CustomFunctionTool(func, description, name, processed_schema)

    def get_tools(
        self,
        actions: t.Optional[t.Sequence[ActionType]] = None,
        apps: t.Optional[t.Sequence[AppType]] = None,
        tags: t.Optional[t.List[TagType]] = None,
        entity_id: t.Optional[str] = None,
        *,
        processors: t.Optional[ProcessorsType] = None,
        skip_default: bool = False,
        check_connected_accounts: bool = True,
    ) -> t.Sequence[FunctionTool]:
        """
        Get composio tools as Autogen FunctionTool objects.

        Args:
            actions: List of actions to wrap
            apps: List of apps to wrap
            tags: Filter apps by given tags
            entity_id: Entity ID for function wrapper
            processors: Optional dict of processors to merge
            check_connected_accounts: Whether to check for connected accounts

        Returns:
            List of Autogen FunctionTool objects
        """
        self.validate_tools(apps=apps, actions=actions, tags=tags)
        if processors is not None:
            self._processor_helpers.merge_processors(processors)

        tools = [
            self._wrap_tool(
                schema=tool.model_dump(exclude_none=True),
                entity_id=entity_id or self.entity_id,
                skip_default=skip_default,
            )
            for tool in self.get_action_schemas(
                actions=actions,
                apps=apps,
                tags=tags,
                check_connected_accounts=check_connected_accounts,
                _populate_requested=True,
            )
        ]

        return tools
