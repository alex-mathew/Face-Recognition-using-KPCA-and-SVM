<pre>
\\\     \\       \\     \      \\
 \\\\    \\\  \\  \\\    \\  \ \\\\    \
  \\\\\  \\\\\ \\  \\\\\  \\\ \\ \\\\   \\  \\
    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  \\  \\
      \\\\\\\\\\ \\\\\\\\\\\\\\\\\\\\\\\\\\\  \\
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\         `
     \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
         \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\      \
            \\\\\\\\\\\\\\\\\\\\\     \\\\\\\\\\\\\\\\\      .
               \\\\\\\\\\                     \\\\\\\\\\\
                  \\        \\\wWWWWWWWww.          \\\\\\\    `
                      \\ \\\WWW"""::::::""WWw         \\\\\    ,
                 \  \\ \\wWWW" .,wWWWWWWw..  WWw.        \\\
              ` ` \\\\\wWW"   W888888888888W  "WXX.       `\\
               . `.\\wWW"   M88888i#####888"8M  "WWX.      `\`
              \ \` wWWW"   M88888##d###"w8oo88M   WWMX.     `\
               ` \wWWW"   :W88888####*  #88888M;   WWIZ.     ``
           - -- wWWWW"     W88888####42##88888W     WWWXx .
               / "WIZ       W8n889######98888W       WWXx.
              ' '/"Wm,       W88888999988888W        >WWR" :
               '   "WMm.      "WW88888888WW"        mmMM" '
                     "Wmm.       "WWWWWW"        ,whAT?"
                      ""MMMmm..            _,mMMMM"""
                           ""MMMMMMMMMMMMMM""""
</pre>




# Setup
1. Make sure your default python installation is Python 3

2. Install virtualenv:

	<code>pip install virtualenv</code>

3. Create a virtualenv named venv:

	<code>virtualenv venv</code>

4. Activate the virtualenv:

	<code>source ./venv/bin/activate</code> {for linux/osx}

	<code>.\venv\Scripts\activate</code> {for win}

4. Install dependencies in virtualenv:

	<code>pip install -r requirements.txt</code>

# Run
 <code>python app.py</code>

# Issues
If some dependency errors occur, try:
<code>pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew kivy.deps.gstreamer</code>