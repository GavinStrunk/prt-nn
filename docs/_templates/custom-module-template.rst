{{ fullname.split('.')[-1] }}
{{ underline }}

.. automodule:: {{ fullname }}

   

   {% block functions %}

   {% if functions %}

   Functions
   ---------

      {% if functions|length > 1 %}

      .. autosummary::
         :nosignatures:
      {% for item in functions %}
         {{ item }}
      {%- endfor %}
      {% endif %}

   {% for item in functions %}

   .. autofunction:: {{ item }}
      
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   Classes
   -------

      {% if classes|length > 1 %}

      .. autosummary::
         :nosignatures:
      {% for item in classes %}
         {{ item }}
      {%- endfor %}
      {% endif %}

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :inherited-members:

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ----------

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}