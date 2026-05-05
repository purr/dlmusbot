from pydantic import TypeAdapter

from spotify_downloader.unplayplay.generated.runtimefunction_data import runtimefunction_data
from spotify_downloader.unplayplay.generated.runtimefunction_models import RuntimeFunction
from spotify_downloader.unplayplay.generated.throwinfo_data import throwinfo_data
from spotify_downloader.unplayplay.generated.throwinfo_models import ThrowInfo
from spotify_downloader.unplayplay.seh.metadata import get_handler_data
from spotify_downloader.unplayplay.seh.state import SehRuntimeState


def build_state(image_base: int) -> SehRuntimeState:
    rf_adapter = TypeAdapter(list[RuntimeFunction])
    ti_adapter = TypeAdapter(list[ThrowInfo])

    runtime_functions_all = rf_adapter.validate_python(runtimefunction_data)
    throw_infos_list = ti_adapter.validate_python(throwinfo_data)

    runtime_functions = sorted(
        (rf for rf in runtime_functions_all if get_handler_data(rf) is not None),
        key=lambda rf: rf.start_rva,
    )

    throw_infos = {ti.struct_rva: ti for ti in throw_infos_list}

    return SehRuntimeState(
        image_base=image_base,
        runtime_functions=runtime_functions,
        throw_infos=throw_infos,
        runtime_function_starts=[rf.start_rva for rf in runtime_functions],
    )
