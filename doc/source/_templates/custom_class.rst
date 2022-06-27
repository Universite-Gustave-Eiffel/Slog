{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


 
{% block attributes %}
{% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: custom_attributes.rst
      :toctree: .
      {% for item in attributes %}
         ~{{ objname }}.{{ item }}
      {%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
      :template: custom_method.rst
      :toctree: .
         {% for item in methods %}
         ~{{ objname }}.{{ item }}
         {%- endfor %}
{% endif %}
{% endblock %}


