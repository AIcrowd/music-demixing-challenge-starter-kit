from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_film import DCUN_TFC_FiLM_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_film_lasaft import DCUN_TFC_FiLM_LaSAFT_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_film_tdf import DCUN_TFC_FiLM_TDF_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm import DCUN_TFC_GPoCM_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lasaft import DCUN_TFC_GPoCM_LaSAFT_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lightsaft import \
    DCUN_TFC_GPoCM_LightSAFT_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_tdf import DCUN_TFC_GPoCM_TDF_Framework

def get_class_by_name(problem_name, model_name):

    if problem_name == 'conditioned_separation':
        if model_name   == 'CUNET_TFC_FiLM':
            return DCUN_TFC_FiLM_Framework
        elif model_name == 'CUNET_TFC_FiLM_TDF':
            return DCUN_TFC_FiLM_TDF_Framework
        elif model_name == 'CUNET_TFC_FiLM_LaSAFT':
            return DCUN_TFC_FiLM_LaSAFT_Framework

        elif model_name == 'CUNET_TFC_GPoCM':
            return DCUN_TFC_GPoCM_Framework
        elif model_name == 'CUNET_TFC_GPoCM_TDF':
            return DCUN_TFC_GPoCM_TDF_Framework
        elif model_name == 'CUNET_TFC_GPoCM_LaSAFT':
            return DCUN_TFC_GPoCM_LaSAFT_Framework
        elif model_name == 'lasaft_net':
            return DCUN_TFC_GPoCM_LaSAFT_Framework
        elif model_name == 'CUNET_TFC_GPoCM_LightSAFT':
            return DCUN_TFC_GPoCM_LightSAFT_Framework
        elif model_name == 'lightsaft_net':
            return DCUN_TFC_GPoCM_LightSAFT_Framework

        else:
            raise NotImplementedError

    elif problem_name == 'dedicated':
        raise NotImplementedError